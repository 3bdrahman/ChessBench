"""Configurable adapter for arbitrary OpenAI-compatible chat endpoints."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn
from urllib.parse import urlparse

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
from chessbench.providers._openai_compat import extract_chat_message, usage_of
from chessbench.providers.registry import register_provider

_log = logging.getLogger(__name__)

_ENV_PREFIX = "OPENAI_COMPATIBLE_"
_ALLOWED_MAX_TOKEN_FIELDS = {"max_tokens", "max_completion_tokens", ""}


def _json_mapping(value: Mapping[str, Any] | str | None, setting: str) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, Mapping):
        return dict(value)
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
    return tuple(dict.fromkeys(item.strip() for item in items if item.strip()))


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


def _normalize_base_url(value: str | None) -> str:
    if not value:
        return ""
    base_url = value.strip().rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "OpenAI-compatible base_url must be an absolute http(s) URL "
            "including its API prefix, for example http://localhost:8000/v1"
        )
    return base_url


def _message_text(message: object) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        text_content = content
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
        text_content = "".join(parts)
    else:
        text_content = ""

    model_extra = getattr(message, "model_extra", None)
    reasoning = (
        getattr(message, "reasoning", None)
        or getattr(message, "reasoning_content", None)
        or (
            model_extra.get("reasoning") or model_extra.get("reasoning_content")
            if isinstance(model_extra, dict)
            else None
        )
    )
    if isinstance(reasoning, str) and reasoning:
        return f"<reasoning>\n{reasoning}\n</reasoning>\n{text_content}"
    return text_content


def _tool_calls(message: object) -> list[dict[str, Any]] | None:
    calls = getattr(message, "tool_calls", None)
    if not calls:
        return None

    output: list[dict[str, Any]] = []
    for call in calls:
        function = getattr(call, "function", None)
        if function is None:
            continue
        arguments = getattr(function, "arguments", "")
        if isinstance(arguments, str):
            with contextlib.suppress(json.JSONDecodeError):
                arguments = json.loads(arguments)
        output.append(
            {
                "name": getattr(function, "name", ""),
                "arguments": arguments,
            }
        )
    return output or None


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
    ) -> None:
        self.base_url = _normalize_base_url(base_url or os.getenv(f"{_ENV_PREFIX}BASE_URL"))
        self.configured_models = _model_ids(
            models if models is not None else os.getenv(f"{_ENV_PREFIX}MODELS")
        )
        self.context_window = int(
            context_window or os.getenv(f"{_ENV_PREFIX}CONTEXT_WINDOW") or DEFAULT_CONTEXT_WINDOW
        )
        self.default_headers = {
            str(name): str(value)
            for name, value in _json_mapping(
                default_headers
                if default_headers is not None
                else os.getenv(f"{_ENV_PREFIX}HEADERS"),
                f"{_ENV_PREFIX}HEADERS",
            ).items()
        }
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
        return AsyncOpenAI(
            api_key=api_key.strip() or "not-needed",
            base_url=self.base_url,
            timeout=timeout or self.timeout,
            default_headers=self.default_headers,
            default_query=self.default_query,
            max_retries=0,
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

        models = [
            ModelInfo(
                id=model.id,
                name=model.id,
                provider=self.name,
                context_window=self.context_window,
                capabilities=[CAP_CHESS],
            )
            for model in response.data
            if getattr(model, "id", None)
        ]
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

        latency_ms = int((time.perf_counter() - start) * 1000)
        message = extract_chat_message(response, self.name, model)
        tokens_in, tokens_out = usage_of(response)
        raw_response = response.model_dump() if hasattr(response, "model_dump") else None
        return CompletionResult(
            text=_message_text(message),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            raw_response=raw_response,
            tool_calls=_tool_calls(message),
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
                host=urlparse(self.base_url).hostname,
            ) from exc
        if isinstance(exc, APIConnectionError):
            raise ConnectionError(
                provider=self.name,
                host=urlparse(self.base_url).hostname,
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


__all__ = ["OpenAICompatibleProvider"]
