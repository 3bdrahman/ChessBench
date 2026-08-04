"""OpenRouter provider implementation (OpenAI-compatible aggregator)."""

from openai import AsyncOpenAI

from .base import ChatMessage, CompletionResult, ModelInfo, ModelProvider
from .registry import register_provider


@register_provider
class OpenRouterProvider(ModelProvider):
    name = "openrouter"
    requires_api_key = True

    def __init__(self):
        self.base_url = "https://openrouter.ai/api/v1"

    def validate_key(self, api_key: str) -> bool:
        return api_key.startswith("sk-or-") and len(api_key) > 20

    async def list_models(self, api_key: str) -> list[ModelInfo]:
        client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)
        models = await client.models.list()

        chat_models = []
        for model in models.data:
            model_id = model.id
            chat_models.append(ModelInfo(
                id=model_id,
                name=model_id,
                provider="openrouter",
                context_window=128000,  # Varies by model
            ))
        return chat_models

    async def complete(self, api_key: str, model: str, messages: list[ChatMessage], **params) -> CompletionResult:
        client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)

        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        temperature = params.get("temperature", 0.1)
        max_tokens = params.get("max_tokens", 100)

        import time
        start = time.time()

        response = await client.chat.completions.create(
            model=model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        latency_ms = int((time.time() - start) * 1000)

        return CompletionResult(
            text=response.choices[0].message.content,
            tokens_in=response.usage.prompt_tokens if response.usage else None,
            tokens_out=response.usage.completion_tokens if response.usage else None,
            latency_ms=latency_ms,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
        )
