"""OpenAI provider implementation."""

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from chess_fight.providers.base import ChatMessage, CompletionResult, ModelInfo, ModelProvider
from chess_fight.providers.registry import register_provider


@register_provider
class OpenAIProvider(ModelProvider):
    name = "openai"
    requires_api_key = True

    def validate_key(self, api_key: str) -> bool:
        return api_key.startswith("sk-") and len(api_key) > 20

    async def list_models(self, api_key: str) -> list[ModelInfo]:
        client = AsyncOpenAI(api_key=api_key)
        models = await client.models.list()

        # Filter for chat models
        chat_models = []
        for model in models.data:
            model_id = model.id
            if any(prefix in model_id for prefix in ["gpt-", "o1-"]) and "embedding" not in model_id and "audio" not in model_id:
                chat_models.append(ModelInfo(
                    id=model_id,
                    name=model_id,
                    provider="openai",
                    context_window=128000 if "gpt-4" in model_id else 8192,
                ))
        return chat_models

    async def complete(self, api_key: str, model: str, messages: list[ChatMessage], **params) -> CompletionResult:
        client = AsyncOpenAI(api_key=api_key)

        # Convert messages
        openai_messages: list[ChatCompletionMessageParam] = [
            {"role": m.role, "content": m.content}  # type: ignore[misc]
            for m in messages
        ]

        # Extract common params
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
            text=response.choices[0].message.content or "",
            tokens_in=response.usage.prompt_tokens if response.usage else None,
            tokens_out=response.usage.completion_tokens if response.usage else None,
            latency_ms=latency_ms,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
        )
