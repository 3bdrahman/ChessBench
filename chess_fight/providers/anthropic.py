"""Anthropic provider implementation."""

import contextlib
import logging
import time
from pathlib import Path
from typing import Any

import yaml
from anthropic import AsyncAnthropic

from chess_fight.common.common_types import (
    CAP_CHESS,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_TEMPERATURE,
    ChatMessage,
    CompletionResult,
    ModelInfo,
    ModelProvider,
)
from chess_fight.common.exceptions import (
    AuthenticationError,
    ConnectionError,
    InvalidApiKeyError,
    ModelNotFoundError,
    ProviderAPIError,
    RateLimitError,
    TimeoutError,
)
from chess_fight.providers.registry import register_provider

_log = logging.getLogger(__name__)

MODELS_CONFIG_PATH = Path(__file__).parent / "models.yaml"


def _load_anthropic_models() -> list[tuple[str, str]]:
    """Load known Anthropic models from YAML config."""
    try:
        with open(MODELS_CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        return [(m[0], m[1]) for m in data.get("ANTHROPIC_MODELS", [])]
    except Exception as exc:
        _log.warning("Failed to load Anthropic models from config: %s", exc)
        return []


_KNOWN_ANTHROPIC_MODELS: list[tuple[str, str]] = _load_anthropic_models()


@register_provider
class AnthropicProvider(ModelProvider):
    name = "anthropic"
    requires_api_key = True

    def __init__(self) -> None:
        pass

    def validate_key(self, api_key: str) -> bool:
        return api_key.startswith("sk-ant-") and len(api_key) > 30

    def _key_prefix_hint(self) -> str:
        return "sk-ant-"

    async def list_models(self, api_key: str) -> list[ModelInfo]:
        client = AsyncAnthropic(
            api_key=api_key,
            timeout=DEFAULT_HTTP_TIMEOUT,
        )
        try:
            # Use the dynamic models.list() endpoint
            models_response = await client.models.list()
            models: list[ModelInfo] = []
            for model in models_response.data:
                model_id = model.id
                # Only include chat models (not embeddings, etc.)
                if model_id.startswith("claude-"):
                    models.append(ModelInfo(
                        id=model_id,
                        name=model.display_name or model_id,
                        provider="anthropic",
                        context_window=DEFAULT_CONTEXT_WINDOW,
                        capabilities=[CAP_CHESS],
                    ))
            if models:
                return models
        except Exception as exc:
            _log.warning("Anthropic list_models failed, falling back to known models: %s", exc)

        # Fallback to curated list if API call fails
        return [
            ModelInfo(
                id=model_id,
                name=name,
                provider="anthropic",
                context_window=DEFAULT_CONTEXT_WINDOW,
                capabilities=[CAP_CHESS],
            )
            for model_id, name in _KNOWN_ANTHROPIC_MODELS
        ]

    async def complete(
        self,
        api_key: str,
        model: str,
        messages: list[ChatMessage],
        **params: Any,
    ) -> CompletionResult:
        client = AsyncAnthropic(
            api_key=api_key,
            timeout=params.get("timeout", DEFAULT_HTTP_TIMEOUT),
        )

        # Anthropic separates system prompts from user/assistant messages.
        system_prompt: str | None = None
        user_messages: list[dict[str, str]] = []
        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            else:
                user_messages.append({"role": m.role, "content": m.content})

        temperature = params.get("temperature", DEFAULT_TEMPERATURE)
        max_tokens = params.get("max_tokens", 100)

        start = time.time()
        try:
            request_kwargs: dict[str, Any] = {
                "model": model,
                "messages": user_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if system_prompt:
                request_kwargs["system"] = system_prompt

            response = await client.messages.create(**request_kwargs)
        except Exception as exc:
            latency_ms = int((time.time() - start) * 1000)
            _classify_and_raise(exc, "anthropic", model, latency_ms, api_key)

        latency_ms = int((time.time() - start) * 1000)

        text = ""
        if response.content:
            first_block = response.content[0]
            if hasattr(first_block, "text"):
                text = first_block.text

        return CompletionResult(
            text=text,
            tokens_in=response.usage.input_tokens if response.usage else None,
            tokens_out=response.usage.output_tokens if response.usage else None,
            latency_ms=latency_ms,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )


def _classify_and_raise(exc: Exception, provider: str, model: str, latency_ms: int, api_key: str = "") -> None:
    """Map Anthropic SDK exceptions to typed ChessFightError and raise."""
    from anthropic import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        NotFoundError,
    )
    from anthropic import (
        AuthenticationError as AnthropicAuthError,
    )
    from anthropic import (
        RateLimitError as AnthropicRateLimitError,
    )

    if isinstance(exc, AnthropicAuthError):
        if getattr(exc, "status_code", 401) == 401:
            raise InvalidApiKeyError(
                provider=provider,
                got_prefix="sk-ant-…" if api_key else "",
                expected_prefix="sk-ant-",
                http_status=401,
            ) from exc
        raise AuthenticationError(
            provider=provider,
            detail=str(exc),
            http_status=exc.status_code or 403,
        ) from exc

    if isinstance(exc, AnthropicRateLimitError):
        retry_after = None
        if hasattr(exc, "response") and exc.response is not None:
            retry_after_hdr = exc.response.headers.get("retry-after")
            if retry_after_hdr:
                with contextlib.suppress(ValueError):
                    retry_after = float(retry_after_hdr)
        raise RateLimitError(
            provider=provider,
            retry_after=retry_after,
            http_status=429,
            raw_response={"error": str(exc), "provider": provider, "model": model},
        ) from exc

    if isinstance(exc, APITimeoutError):
        raise TimeoutError(
            provider=provider,
            timeout_seconds=30.0,
        ) from exc

    if isinstance(exc, APIConnectionError):
        raise ConnectionError(
            provider=provider,
            host="api.anthropic.com",
            detail=str(exc),
        ) from exc

    if isinstance(exc, NotFoundError):
        raise ModelNotFoundError(
            provider=provider,
            model_id=model,
            available_models=[m[0] for m in _KNOWN_ANTHROPIC_MODELS],
        ) from exc

    if isinstance(exc, InternalServerError):
        raise ProviderAPIError(
            provider=provider,
            status_code=500,
            detail=str(exc),
            raw_response={"error": str(exc), "provider": provider, "model": model},
        ) from exc

    status_code = getattr(exc, "status_code", None)
    if status_code and 500 <= status_code < 600:
        raise ProviderAPIError(
            provider=provider,
            status_code=status_code,
            detail=str(exc),
            raw_response={"error": str(exc), "provider": provider, "model": model},
        ) from exc

    raise ProviderAPIError(
        provider=provider,
        status_code=status_code or 500,
        detail=str(exc),
        raw_response={"error": str(exc), "provider": provider, "model": model},
    ) from exc
