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
        import httpx
        from chess_fight.common.common_types import _NON_CHAT_TOKENS, _WEAK_FOR_CHESS_TOKENS

        try:
            async with httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {api_key}"}
                )
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            _log.warning("OpenRouter list_models failed: %s", exc)
            return []

        chat_models: list[ModelInfo] = []
        for model in data.get("data", []):
            model_id = model["id"]
            mid = model_id.lower()

            # First check token filters
            if any(tok in mid for tok in _NON_CHAT_TOKENS):
                continue
            
            # Check architecture to ensure it's a text-output model
            arch = model.get("architecture") or {}
            out_mods = arch.get("output_modalities") or []
            in_mods = arch.get("input_modalities") or []
            
            # Must be capable of taking text and outputting text
            if "text" not in out_mods or "text" not in in_mods:
                continue

            capabilities = []
            if not any(tok in mid for tok in _WEAK_FOR_CHESS_TOKENS):
                capabilities.append(CAP_CHESS)
            
            is_free = ":free" in mid or mid.endswith(":free")
            name = model_id
            if is_free:
                name = f"{model_id} ★free"
                
            ctx = model.get("context_length") or DEFAULT_CONTEXT_WINDOW

            chat_models.append(ModelInfo(
                id=model_id,
                name=name,
                provider="openrouter",
                context_window=ctx,
                pricing_tier="free" if is_free else "paid",
                capabilities=capabilities,
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
            default_headers={
                "HTTP-Referer": "https://github.com/3bdrahman/chess_fight",
                "X-Title": "Chess Fight",
            }
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
            "extra_body": {"include_reasoning": True},
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
            _classify_and_raise(exc, "openrouter", model, latency_ms, api_key)

        latency_ms = int((time.time() - start) * 1000)

        msg = response.choices[0].message
        content = msg.content or ""
        
        # If it's a reasoning model, append the reasoning to the content so our parser can log it
        # and extract moves from it if the model put the move inside the reasoning block.
        reasoning = getattr(msg, "reasoning", None)
        if reasoning is None and getattr(msg, "model_extra", None) and isinstance(msg.model_extra, dict):
            reasoning = msg.model_extra.get("reasoning")
            
        if reasoning and isinstance(reasoning, str):
            content = f"<reasoning>\n{reasoning}\n</reasoning>\n{content}"

        tool_calls_out = None
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            import json
            tool_calls_out = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = tc.function.arguments
                tool_calls_out.append({
                    "name": tc.function.name,
                    "arguments": args
                })

        return CompletionResult(
            text=content,
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
    if status_code == 402:
        from chess_fight.common.exceptions import QuotaExceededError
        raise QuotaExceededError(
            provider=provider,
            detail=str(exc),
        ) from exc

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
