import inspect
import logging
import time
from typing import Any

from google import genai
from google.genai import types

from chess_fight.common.common_types import (
    CAP_CHESS,
    DEFAULT_CONTEXT_WINDOW,
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
class GoogleProvider(ModelProvider):
    name = "google"
    requires_api_key = True

    def __init__(self) -> None:
        pass

    def validate_key(self, api_key: str) -> bool:
        return len(api_key) > 20

    def _key_prefix_hint(self) -> str:
        return "Google AI key"

    async def list_models(self, api_key: str) -> list[ModelInfo]:
        client = genai.Client(api_key=api_key)
        try:
            models_response = client.models.list()
            # Handle both sync and async versions
            if inspect.iscoroutine(models_response):
                models_response = await models_response
        except Exception as exc:
            _log.warning("Google list_models failed: %s", exc)
            return []

        chat_models: list[ModelInfo] = []
        for model in models_response:
            supported = list(model.supported_actions or [])
            if "generateContent" not in supported:
                continue
            if model.name is None:
                continue
            mid_full = model.name.lower()
            if "gemini" not in mid_full:
                continue
            model_id = model.name.replace("models/", "")
            mid = model_id.lower()
            if any(tok in mid for tok in ("embedding", "imagen", "veo", "aqa")):
                continue

            chat_models.append(ModelInfo(
                id=model_id,
                name=model_id,
                provider="google",
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
        client = genai.Client(api_key=api_key)

        # Build a proper multi-turn conversation history so the model sees prior
        # assistant moves. We collapse the chat history into a single generate_content
        # call using a system instruction + contents list.
        system_prompt = ""
        contents: list[types.Content] = []
        for m in messages:
            if m.role == "system":
                # Multiple system messages get concatenated.
                system_prompt = (system_prompt + "\n\n" + m.content).strip() if system_prompt else m.content
            elif m.role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=m.content)]))
            elif m.role == "assistant":
                # FIX: previously these were silently dropped, breaking multi-turn context.
                contents.append(types.Content(role="model", parts=[types.Part(text=m.content)]))

        temperature = params.get("temperature", DEFAULT_TEMPERATURE)
        max_tokens = params.get("max_tokens", 100)

        start = time.time()
        try:
            config_kwargs: dict[str, object] = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt

            response = await client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),  # type: ignore[arg-type]
            )
        except Exception as exc:
            latency_ms = int((time.time() - start) * 1000)
            _classify_and_raise(exc, "google", model, latency_ms, api_key)

        latency_ms = int((time.time() - start) * 1000)

        tokens_in = None
        tokens_out = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens_in = response.usage_metadata.prompt_token_count
            tokens_out = response.usage_metadata.candidates_token_count

        return CompletionResult(
            text=response.text if response.text else "",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )


def _classify_and_raise(exc: Exception, provider: str, model: str, latency_ms: int, api_key: str = "") -> None:
    """Map google-genai exceptions to typed ChessFightError and raise."""
    # google-genai uses exceptions with .code attribute (gRPC status codes)
    # Common codes: 14=UNAVAILABLE, 4=DEADLINE_EXCEEDED, 7=PERMISSION_DENIED, 5=NOT_FOUND
    code = getattr(exc, "code", None)
    status_code = getattr(exc, "status_code", None)

    # Some google-genai errors wrap gRPC errors with .details()
    detail = str(exc)

    if code == 7 or status_code == 403:  # PERMISSION_DENIED
        raise AuthenticationError(
            provider=provider,
            detail=detail,
            http_status=403,
        ) from exc

    if code == 16 or status_code == 401:  # UNAUTHENTICATED
        raise InvalidApiKeyError(
            provider=provider,
            got_prefix=api_key[:8] + "…" if api_key else "",
            expected_prefix="Google AI key",
            http_status=401,
        ) from exc

    if code == 8 or status_code == 429:  # RESOURCE_EXHAUSTED
        raise RateLimitError(
            provider=provider,
            retry_after=None,
            http_status=429,
            raw_response={"error": detail, "provider": provider, "model": model},
        ) from exc

    if code == 4 or status_code == 504:  # DEADLINE_EXCEEDED
        raise TimeoutError(
            provider=provider,
            timeout_seconds=30.0,
        ) from exc

    if code == 14 or (status_code and 500 <= status_code < 600):  # UNAVAILABLE or 5xx
        raise ProviderAPIError(
            provider=provider,
            status_code=status_code or 503,
            detail=detail,
            raw_response={"error": detail, "provider": provider, "model": model},
        ) from exc

    if code == 5 or status_code == 404:  # NOT_FOUND
        raise ModelNotFoundError(
            provider=provider,
            model_id=model,
            available_models=None,
        ) from exc

    if code == 2 or code == 10:  # UNKNOWN or INTERNAL
        raise ProviderAPIError(
            provider=provider,
            status_code=500,
            detail=detail,
            raw_response={"error": detail, "provider": provider, "model": model},
        ) from exc

    # Connection errors
    if "connection" in detail.lower() or "connect" in detail.lower():
        raise ConnectionError(
            provider=provider,
            host="generativelanguage.googleapis.com",
            detail=detail,
        ) from exc

    # Non-retryable catch-all
    raise ProviderAPIError(
        provider=provider,
        status_code=status_code or 500,
        detail=detail,
        raw_response={"error": detail, "provider": provider, "model": model},
    ) from exc
