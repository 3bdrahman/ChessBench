"""Groq provider implementation (OpenAI-compatible API).

Groq offers a generous free tier with fast inference, making it ideal
for hosted demos where users don't need to provide their own API keys.
"""

import contextlib
import logging
import os
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

# Rate-limited free models known to work well for chess moves
_GROQ_FREE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "llama3-70b-8192",
    "llama3-8b-8192",
]


@register_provider
class GroqProvider(ModelProvider):
    name = "groq"
    requires_api_key = True

    def __init__(self) -> None:
        self.base_url = os.getenv(
            "GROQ_BASE_URL",
            "https://api.groq.com/openai/v1",
        )

    def validate_key(self, api_key: str) -> bool:
        return api_key.startswith("gsk_") and len(api_key) > 20

    def _key_prefix_hint(self) -> str:
        return "gsk_"

    async def list_models(self, api_key: str) -> list[ModelInfo]:
        client = AsyncOpenAI(api_key=api_key, base_url=self.base_url, timeout=DEFAULT_HTTP_TIMEOUT)
        try:
            models = await client.models.list()
        except Exception as exc:
            _log.warning("Groq list_models failed: %s", exc)
            return []

        chat_models: list[ModelInfo] = []
        seen: set[str] = set()
        for model in models.data:
            model_id = model.id
            if model_id in seen:
                continue
            seen.add(model_id)

            mid = model_id.lower()
            # Groq exposes some preview/whisper/guard models in the same endpoint.
            if any(tok in mid for tok in (
                "whisper", "tts", "guard", "safeguard", "tool-use",
                "preview", "composite",
            )):
                continue

            if any(
                prefix in mid
                for prefix in [
                    "llama", "mixtral", "gemma", "deepseek",
                    "qwen", "mistral",
                ]
            ):
                is_free = model_id in _GROQ_FREE_MODELS
                chat_models.append(
                    ModelInfo(
                        id=model_id,
                        name=model_id + (" ★free" if is_free else ""),
                        provider="groq",
                        context_window=DEFAULT_CONTEXT_WINDOW,
                        pricing_tier="free" if is_free else "paid",
                        capabilities=[CAP_CHESS],
                    )
                )
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
            _classify_and_raise(exc, "groq", model, latency_ms, api_key)

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
                got_prefix="gsk_…" if api_key else "",
                expected_prefix="gsk_",
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
            host="api.groq.com",
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
