"""NVIDIA NIM provider implementation (OpenAI-compatible)."""

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


@register_provider
class NIMProvider(ModelProvider):
    name = "nim"
    requires_api_key = True

    def __init__(self) -> None:
        self.base_url = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

    def validate_key(self, api_key: str) -> bool:
        return len(api_key) >= 10

    def _key_prefix_hint(self) -> str:
        return "NIM key"

    async def list_models(self, api_key: str) -> list[ModelInfo]:
        verified_models = [
            "google/gemma-4-31b-it",
            "nvidia/nemotron-3.5-lightning-30b-a3b",
            "nvidia/nemotron-3-super-120b-a12b",
            "stepfun-ai/step-3.7-flash",
            "deepseek-ai/deepseek-v4-flash-0731",
            "mistralai/mistral-nemotron"
        ]
        
        chat_models: list[ModelInfo] = []
        for model_id in verified_models:
            chat_models.append(ModelInfo(
                id=model_id,
                name=model_id,
                provider="nim",
                context_window=DEFAULT_CONTEXT_WINDOW,
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
        max_tokens = params.get("max_tokens")

        completion_kwargs: dict[str, Any] = {
            "model": model,
            "messages": openai_messages,  # type: ignore[arg-type]
            "temperature": temperature,
        }
        if max_tokens is not None:
            completion_kwargs["max_tokens"] = max_tokens

        if "tools" in params:
            completion_kwargs["tools"] = params["tools"]
            if "tool_choice" in params:
                completion_kwargs["tool_choice"] = params["tool_choice"]

        start = time.time()
        try:
            response = await client.chat.completions.create(**completion_kwargs)
        except Exception as exc:
            latency_ms = int((time.time() - start) * 1000)
            _classify_and_raise(exc, "nim", model, latency_ms, api_key)

        latency_ms = int((time.time() - start) * 1000)

        tool_calls_out = None
        message = response.choices[0].message
        if hasattr(message, "tool_calls") and message.tool_calls:
            import json
            tool_calls_out = []
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = tc.function.arguments
                tool_calls_out.append({
                    "name": tc.function.name,
                    "arguments": args
                })

        return CompletionResult(
            text=message.content or "",
            tokens_in=response.usage.prompt_tokens if response.usage else None,
            tokens_out=response.usage.completion_tokens if response.usage else None,
            latency_ms=latency_ms,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            tool_calls=tool_calls_out,
        )


def _classify_and_raise(exc: Exception, provider: str, model: str, latency_ms: int, api_key: str = "") -> None:
    """Map OpenAI SDK exceptions to typed ChessFightError and raise."""
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        NotFoundError,
        PermissionDeniedError,
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
                got_prefix=api_key[:8] + "…" if api_key else "",
                expected_prefix="NIM key",
                http_status=401,
            ) from exc
        raise AuthenticationError(
            provider=provider,
            detail=str(exc),
            http_status=exc.status_code or 403,
        ) from exc

    if isinstance(exc, PermissionDeniedError):
        # 403 = valid key format but not authorized (expired, no access to model, etc.)
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
            host="integrate.api.nvidia.com",
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
