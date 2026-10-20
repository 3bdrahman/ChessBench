"""Configurable adapter for arbitrary OpenAI-compatible chat endpoints."""

from __future__ import annotations

import contextlib
import ipaddress
import json
import logging
import os
import re
import socket
import time
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx
from openai import AsyncOpenAI

from chessbench.common.common_types import (
    CAP_CHESS,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_TEMPERATURE,
    ChatMessage,
    CompletionResult,
    ModelInfo,
    ModelProvider,
)
from chessbench.common.exceptions import (
    AuthenticationError,
    ConnectionError,
    ModelNotFoundError,
    ProviderAPIError,
    RateLimitError,
    TimeoutError,
)
from chessbench.providers._openai_compat import (
    extract_chat_message,
    message_text,
    tool_calls_of,
    usage_of,
)
from chessbench.providers.registry import register_provider

_log = logging.getLogger(__name__)

_ENV_PREFIX = "OPENAI_COMPATIBLE_"
_ALLOWED_MAX_TOKEN_FIELDS = {"max_tokens", "max_completion_tokens", ""}
_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_FORBIDDEN_HEADERS = {"content-length", "host", "transfer-encoding"}
_RESOURCE_SUFFIXES = ("/chat/completions", "/completions", "/models")
_MAX_CONFIG_JSON_CHARS = 32_768
_MAX_HEADER_VALUE_CHARS = 8_192
_MAX_MODELS = 1_000
_MAX_MODEL_ID_CHARS = 512
_MAX_COMPLETION_CHARS = 200_000


def _json_mapping(value: Mapping[str, Any] | str | None, setting: str) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if len(value) > _MAX_CONFIG_JSON_CHARS:
        raise ValueError(f"{setting} exceeds the {_MAX_CONFIG_JSON_CHARS}-character limit")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{setting} must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{setting} must be a JSON object")
    return parsed


def _model_ids(value: Sequence[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    items = value.split(",") if isinstance(value, str) else value
    normalized: list[str] = []
    for raw_item in items:
        item = str(raw_item).strip()
        if not item:
            continue
        if len(item) > _MAX_MODEL_ID_CHARS or any(ord(char) < 32 for char in item):
            raise ValueError("Model IDs must be printable and at most 512 characters")
        normalized.append(item)
        if len(normalized) > _MAX_MODELS:
            raise ValueError(f"At most {_MAX_MODELS} model IDs may be configured")
    return tuple(dict.fromkeys(normalized))


def _bool_setting(value: bool | str | None, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def normalize_base_url(value: str | None) -> str:
    """Return a canonical API root or reject an ambiguous/malicious URL."""
    if not value:
        return ""
    if len(value) > 2048 or any(ord(character) < 32 for character in value):
        raise ValueError("OpenAI-compatible base_url contains invalid characters")
    if value != value.strip() or any(character.isspace() for character in value):
        raise ValueError("OpenAI-compatible base_url must not contain whitespace")

    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            "OpenAI-compatible base_url must be an absolute http(s) URL "
            "including its API prefix, for example http://localhost:8000/v1"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("OpenAI-compatible base_url must not contain user info")
    if parsed.query or parsed.fragment:
        raise ValueError(
            "OpenAI-compatible base_url must not contain a query or fragment; "
            "use OPENAI_COMPATIBLE_QUERY for query parameters"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("OpenAI-compatible base_url contains an invalid port") from exc

    hostname = parsed.hostname.rstrip(".").lower()
    if not hostname or "%" in hostname:
        raise ValueError("OpenAI-compatible base_url contains an invalid host")
    try:
        canonical_host = hostname.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("OpenAI-compatible base_url contains an invalid host") from exc

    decoded_path = unquote(parsed.path)
    if "\\" in parsed.path or "\\" in decoded_path:
        raise ValueError("OpenAI-compatible base_url path must not contain backslashes")
    if any(segment in {".", ".."} for segment in decoded_path.split("/")):
        raise ValueError("OpenAI-compatible base_url path must not contain dot segments")
    path = parsed.path.rstrip("/")
    if path.lower().endswith(_RESOURCE_SUFFIXES):
        raise ValueError(
            "OpenAI-compatible base_url must be an API root, not a models or "
            "completion resource URL"
        )

    try:
        address = ipaddress.ip_address(canonical_host)
    except ValueError:
        host_for_url = canonical_host
    else:
        host_for_url = f"[{canonical_host}]" if address.version == 6 else canonical_host
    netloc = f"{host_for_url}:{port}" if port is not None else host_for_url
    return urlunsplit((parsed.scheme.lower(), netloc, path, "", ""))


def validate_endpoint_url(
    value: str,
    *,
    enforce_public_networks: bool = False,
    allow_private_networks: bool = False,
    allowed_hosts: Sequence[str] | str | None = None,
    require_https: bool = False,
) -> str:
    """Validate URL shape and, when requested, its resolved network targets."""
    base_url = normalize_base_url(value)
    if not base_url:
        raise ValueError("OpenAI-compatible base_url is required")
    parsed = urlsplit(base_url)
    hostname = parsed.hostname or ""
    if require_https and parsed.scheme != "https":
        raise ValueError("This endpoint requires HTTPS")

    allowed = {host.lower().rstrip(".") for host in _model_ids(allowed_hosts)}
    if allowed and hostname.lower().rstrip(".") not in allowed:
        raise ValueError("Endpoint host is not in OPENAI_COMPATIBLE_ALLOWED_HOSTS")

    if not enforce_public_networks or allow_private_networks:
        return base_url

    if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
        raise ValueError("Private/local endpoints are disabled on public deployments")
    try:
        literal_address = ipaddress.ip_address(hostname)
    except ValueError:
        literal_address = None
    if literal_address is not None and not literal_address.is_global:
        raise ValueError("Private/local endpoints are disabled on public deployments")

    try:
        resolved = {
            item[4][0]
            for item in socket.getaddrinfo(
                hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        }
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve endpoint host: {hostname}") from exc
    if not resolved or any(not ipaddress.ip_address(address).is_global for address in resolved):
        raise ValueError("Private/local endpoints are disabled on public deployments")
    return base_url


def _validated_headers(
    value: Mapping[str, Any] | str | None,
) -> dict[str, str]:
    headers = _json_mapping(value, f"{_ENV_PREFIX}HEADERS")
    output: dict[str, str] = {}
    for raw_name, raw_value in headers.items():
        name = str(raw_name)
        header_value = str(raw_value)
        if not _HEADER_NAME.fullmatch(name) or name.lower() in _FORBIDDEN_HEADERS:
            raise ValueError(f"Unsafe custom HTTP header name: {name!r}")
        if len(header_value) > _MAX_HEADER_VALUE_CHARS or any(
            character in header_value for character in ("\r", "\n", "\0")
        ):
            raise ValueError(f"Unsafe custom HTTP header value for {name!r}")
        output[name] = header_value
    return output


@register_provider
class OpenAICompatibleProvider(ModelProvider):
    """OpenAI Chat Completions adapter configured at runtime.

    The endpoint must include the API prefix expected by the server (usually
    ``/v1``). API keys are optional because many local servers do not require
    authentication. When an endpoint does not implement ``GET /models``, set
    ``models`` or ``OPENAI_COMPATIBLE_MODELS`` to a comma-separated list.
    """

    name = "openai_compatible"
    requires_api_key = False

    def __init__(
        self,
        *,
        base_url: str | None = None,
        models: Sequence[str] | str | None = None,
        context_window: int | None = None,
        default_headers: Mapping[str, Any] | str | None = None,
        default_query: Mapping[str, Any] | str | None = None,
        extra_body: Mapping[str, Any] | str | None = None,
        supports_tools: bool | str | None = None,
        send_temperature: bool | str | None = None,
        max_tokens_field: str | None = None,
        timeout: float | None = None,
        probe_timeout: float | None = None,
        enforce_public_networks: bool | str | None = None,
        allow_private_networks: bool | str | None = None,
        allowed_hosts: Sequence[str] | str | None = None,
        trust_env: bool | str | None = None,
    ) -> None:
        self.enforce_public_networks = _bool_setting(
            enforce_public_networks
            if enforce_public_networks is not None
            else os.getenv(f"{_ENV_PREFIX}ENFORCE_PUBLIC_NETWORKS"),
            False,
        )
        self.allow_private_networks = _bool_setting(
            allow_private_networks
            if allow_private_networks is not None
            else os.getenv(f"{_ENV_PREFIX}ALLOW_PRIVATE_NETWORKS"),
            False,
        )
        self.allowed_hosts = _model_ids(
            allowed_hosts if allowed_hosts is not None else os.getenv(f"{_ENV_PREFIX}ALLOWED_HOSTS")
        )
        self.trust_env = _bool_setting(
            trust_env
            if trust_env is not None
            else os.getenv(f"{_ENV_PREFIX}TRUST_ENV"),
            False,
        )
        raw_base_url = base_url or os.getenv(f"{_ENV_PREFIX}BASE_URL")
        self.base_url = (
            validate_endpoint_url(
                raw_base_url,
                enforce_public_networks=self.enforce_public_networks,
                allow_private_networks=self.allow_private_networks,
                allowed_hosts=self.allowed_hosts,
                require_https=(
                    self.enforce_public_networks and not self.allow_private_networks
                ),
            )
            if raw_base_url
            else ""
        )
        self.configured_models = _model_ids(
            models if models is not None else os.getenv(f"{_ENV_PREFIX}MODELS")
        )
        self.context_window = int(
            context_window or os.getenv(f"{_ENV_PREFIX}CONTEXT_WINDOW") or DEFAULT_CONTEXT_WINDOW
        )
        self.default_headers = _validated_headers(
            default_headers if default_headers is not None else os.getenv(f"{_ENV_PREFIX}HEADERS")
        )
        self.default_query = _json_mapping(
            default_query if default_query is not None else os.getenv(f"{_ENV_PREFIX}QUERY"),
            f"{_ENV_PREFIX}QUERY",
        )
        self.extra_body = _json_mapping(
            extra_body if extra_body is not None else os.getenv(f"{_ENV_PREFIX}EXTRA_BODY"),
            f"{_ENV_PREFIX}EXTRA_BODY",
        )
        self.supports_tools = _bool_setting(
            supports_tools
            if supports_tools is not None
            else os.getenv(f"{_ENV_PREFIX}SUPPORTS_TOOLS"),
            False,
        )
        self.send_temperature = _bool_setting(
            send_temperature
            if send_temperature is not None
            else os.getenv(f"{_ENV_PREFIX}SEND_TEMPERATURE"),
            True,
        )
        self.max_tokens_field = (
            max_tokens_field
            if max_tokens_field is not None
            else os.getenv(f"{_ENV_PREFIX}MAX_TOKENS_FIELD", "max_tokens")
        ).strip()
        if self.max_tokens_field not in _ALLOWED_MAX_TOKEN_FIELDS:
            raise ValueError(
                "max_tokens_field must be 'max_tokens', 'max_completion_tokens', or an empty string"
            )
        self.timeout = float(timeout or os.getenv(f"{_ENV_PREFIX}TIMEOUT") or DEFAULT_HTTP_TIMEOUT)
        self.probe_timeout = float(
            probe_timeout or os.getenv(f"{_ENV_PREFIX}PROBE_TIMEOUT") or 10.0
        )

    def validate_key(self, api_key: str) -> bool:
        return True

    def _key_prefix_hint(self) -> str:
        return "endpoint API key (optional)"

    def _client(self, api_key: str, timeout: float | None = None) -> AsyncOpenAI:
        if not self.base_url:
            raise ProviderAPIError(
                provider=self.name,
                status_code=400,
                detail=(
                    "No endpoint configured. Set OPENAI_COMPATIBLE_BASE_URL "
                    "to the server's API root, including /v1 when required."
                ),
            )
        self.base_url = validate_endpoint_url(
            self.base_url,
            enforce_public_networks=self.enforce_public_networks,
            allow_private_networks=self.allow_private_networks,
            allowed_hosts=self.allowed_hosts,
            require_https=(
                self.enforce_public_networks and not self.allow_private_networks
            ),
        )
        normalized_key = api_key.strip()
        if len(normalized_key) > _MAX_HEADER_VALUE_CHARS or any(
            character in normalized_key for character in ("\r", "\n", "\0")
        ):
            raise ValueError("API key contains invalid characters or is too long")
        return AsyncOpenAI(
            api_key=normalized_key or "not-needed",
            base_url=self.base_url,
            timeout=timeout or self.timeout,
            default_headers=self.default_headers,
            default_query=self.default_query,
            max_retries=0,
            http_client=httpx.AsyncClient(
                follow_redirects=False,
                trust_env=self.trust_env,
            ),
        )

    def _configured_model_info(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id=model_id,
                name=model_id,
                provider=self.name,
                context_window=self.context_window,
                capabilities=[CAP_CHESS],
            )
            for model_id in self.configured_models
        ]

    async def list_models(self, api_key: str) -> list[ModelInfo]:
        client = self._client(api_key, self.probe_timeout)
        try:
            response = await client.models.list()
        except Exception as exc:
            if self.configured_models:
                _log.info(
                    "Compatible endpoint has no usable /models route; using configured models: %s",
                    ", ".join(self.configured_models),
                )
                return self._configured_model_info()
            raise ProviderAPIError(
                provider=self.name,
                status_code=getattr(exc, "status_code", 502) or 502,
                detail=(
                    f"Model discovery failed at {self.base_url}/models: {exc}. "
                    "Set OPENAI_COMPATIBLE_MODELS when the endpoint does not "
                    "implement GET /models."
                ),
                raw_response={"error": str(exc), "base_url": self.base_url},
            ) from exc
        finally:
            with contextlib.suppress(Exception):
                await client.close()

        models: list[ModelInfo] = []
        for model in response.data[:_MAX_MODELS]:
            model_id = getattr(model, "id", None)
            if (
                not isinstance(model_id, str)
                or not model_id
                or len(model_id) > _MAX_MODEL_ID_CHARS
                or any(ord(char) < 32 for char in model_id)
            ):
                continue
            models.append(
                ModelInfo(
                    id=model_id,
                    name=model_id,
                    provider=self.name,
                    context_window=self.context_window,
                    capabilities=[CAP_CHESS],
                )
            )
        return models or self._configured_model_info()

    async def complete(
        self,
        api_key: str,
        model: str,
        messages: list[ChatMessage],
        **params: Any,
    ) -> CompletionResult:
        timeout = float(params.get("timeout", self.timeout))
        client = self._client(api_key, timeout)
        request: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
        }

        if self.send_temperature:
            request["temperature"] = params.get("temperature", DEFAULT_TEMPERATURE)

        max_tokens = params.get("max_tokens")
        if max_tokens is not None and self.max_tokens_field:
            request[self.max_tokens_field] = max_tokens

        if self.supports_tools and "tools" in params:
            request["tools"] = params["tools"]
            if "tool_choice" in params:
                request["tool_choice"] = params["tool_choice"]

        extra_body = dict(self.extra_body)
        request_extra = params.get("extra_body")
        if isinstance(request_extra, Mapping):
            extra_body.update(request_extra)
        if extra_body:
            request["extra_body"] = extra_body

        start = time.perf_counter()
        try:
            response = await client.chat.completions.create(**request)
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            self._raise_api_error(exc, model, api_key, timeout, latency_ms)
        finally:
            with contextlib.suppress(Exception):
                await client.close()

        latency_ms = int((time.perf_counter() - start) * 1000)
        message = extract_chat_message(response, self.name, model)
        tokens_in, tokens_out = usage_of(response)
        text = message_text(message)
        if len(text) > _MAX_COMPLETION_CHARS:
            raise ProviderAPIError(
                provider=self.name,
                status_code=502,
                detail="Completion response exceeded the safe size limit",
            )
        raw_response = {
            "id": str(getattr(response, "id", ""))[:256],
            "model": str(getattr(response, "model", model))[:_MAX_MODEL_ID_CHARS],
            "finish_reason": str(getattr(response.choices[0], "finish_reason", ""))[:128],
        }
        return CompletionResult(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            raw_response=raw_response,
            tool_calls=tool_calls_of(message),
        )

    def _raise_api_error(
        self,
        exc: Exception,
        model: str,
        api_key: str,
        timeout: float,
        latency_ms: int,
    ) -> NoReturn:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            NotFoundError,
            PermissionDeniedError,
        )
        from openai import (
            AuthenticationError as OpenAIAuthenticationError,
        )
        from openai import (
            RateLimitError as OpenAIRateLimitError,
        )

        if isinstance(exc, (OpenAIAuthenticationError, PermissionDeniedError)):
            raise AuthenticationError(
                provider=self.name,
                detail=str(exc),
                http_status=getattr(exc, "status_code", None) or 401,
            ) from exc
        if isinstance(exc, OpenAIRateLimitError):
            retry_after = None
            response = getattr(exc, "response", None)
            if response is not None:
                value = response.headers.get("retry-after")
                if value:
                    with contextlib.suppress(ValueError):
                        retry_after = float(value)
            raise RateLimitError(
                provider=self.name,
                retry_after=retry_after,
                raw_response={"error": str(exc), "model": model},
            ) from exc
        if isinstance(exc, APITimeoutError):
            raise TimeoutError(
                provider=self.name,
                timeout_seconds=timeout,
                host=urlsplit(self.base_url).hostname,
            ) from exc
        if isinstance(exc, APIConnectionError):
            raise ConnectionError(
                provider=self.name,
                host=urlsplit(self.base_url).hostname,
                detail=str(exc),
            ) from exc
        if isinstance(exc, NotFoundError):
            raise ModelNotFoundError(provider=self.name, model_id=model) from exc

        status_code = getattr(exc, "status_code", None) or 500
        raise ProviderAPIError(
            provider=self.name,
            status_code=status_code,
            detail=str(exc),
            raw_response={
                "error": str(exc),
                "model": model,
                "base_url": self.base_url,
                "latency_ms": latency_ms,
                "authenticated": bool(api_key),
            },
        ) from exc


__all__ = [
    "OpenAICompatibleProvider",
    "normalize_base_url",
    "validate_endpoint_url",
]
