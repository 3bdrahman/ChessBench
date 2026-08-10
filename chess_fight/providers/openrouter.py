"""OpenRouter provider implementation (OpenAI-compatible aggregator)."""

import contextlib
import logging
import time
from typing import Any

from openai import AsyncOpenAI

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


@register_provider
class OpenRouterProvider(ModelProvider):
    name = "openrouter"
    requires_api_key = True

    def __init__(self) -> None:
        self.base_url = "https://openrouter.ai/api/v1"

    def validate_key(self, api_key: str) -> bool:
        return api_key.startswith("sk-or-") and len(api_key) > 20

    def _key_prefix_hint(self) -> str:
        return "sk-or-"

    async def list_models(self, api_key: str) -> list[ModelInfo]:
        client = AsyncOpenAI(api_key=api_key, base_url=self.base_url, timeout=DEFAULT_HTTP_TIMEOUT)
        try:
            models = await client.models.list()
        except Exception as exc:
            _log.warning("OpenRouter list_models failed: %s", exc)
            return []

        chat_models: list[ModelInfo] = []
        for model in models.data:
            model_id = model.id
            mid = model_id.lower()
            # OpenRouter mixes image, audio and embedding models into the same
            # /models endpoint — drop anything that obviously isn't chat.
            if any(tok in mid for tok in (
                "embed", "whisper", "tts", "dall-e", "dalle",
                "moderation", "clip", "rerank", "imagen",
                "sdxl", "flux", "sd-", "vision-image", "image-gen",
                "guard", "asr",
            )):
                continue

            is_free = ":free" in mid or mid.endswith(":free")
            name = model_id
            if is_free:
                name = f"{model_id} ★free"

            chat_models.append(ModelInfo(
                id=model_id,
                name=name,
                provider="openrouter",
                context_window=DEFAULT_CONTEXT_WINDOW,
                pricing_tier="free" if is_free else "paid",
                capabilities=[CAP_CHESS],
            ))
        return chat_models

    async def complete(
        self,
        api_key: str,
        model: str,
        messages: list[ChatMessage],
        **params: Any,
    ) -> CompletionResult:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=params.get("timeout", DEFAULT_HTTP_TIMEOUT),
        )

        openai_messages: list[dict[str, str]] = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        temperature = params.get("temperature", DEFAULT_TEMPERATURE)
        max_tokens = params.get("max_tokens", 100)

        start = time.time()
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=openai_messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            latency_ms = int((time.time() - start) * 1000)
            _classify_and_raise(exc, "openrouter", model, latency_ms, api_key)

        latency_ms = int((time.time() - start) * 1000)

        return CompletionResult(
            text=response.choices[0].message.content or "",
            tokens_in=response.usage.prompt_tokens if response.usage else None,
            tokens_out=response.usage.completion_tokens if response.usage else None,
            latency_ms=latency_ms,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )


def _classify_and_raise(exc: Exception, provider: str, model: str, latency_ms: int, api_key: str = "") -> None:
    """Map OpenAI SDK exceptions to typed ChessFightError and raise."""
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        NotFoundError,
    )
    from openai import (
        AuthenticationError as OpenAIAuthError,
    )
    from openai import (
        RateLimitError as OpenAIRateLimitError,
    )

    if isinstance(exc, OpenAIAuthError):
        if getattr(exc, "status_code", 401) == 401:
            raise InvalidApiKeyError(
                provider=provider,
                got_prefix="sk-or-…" if api_key else "",
                expected_prefix="sk-or-",
                http_status=401,
            ) from exc
        raise AuthenticationError(
            provider=provider,
            detail=str(exc),
            http_status=exc.status_code or 403,
        ) from exc

    if isinstance(exc, OpenAIRateLimitError):
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
            host="openrouter.ai",
            detail=str(exc),
        ) from exc

    if isinstance(exc, NotFoundError):
        raise ModelNotFoundError(
            provider=provider,
            model_id=model,
            available_models=None,
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
