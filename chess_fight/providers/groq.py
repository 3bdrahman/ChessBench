"""Groq provider implementation (OpenAI-compatible API).

Groq offers a generous free tier with fast inference, making it ideal
for hosted demos where users don't need to provide their own API keys.
"""

import os
import time

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from chess_fight.providers.base import ChatMessage, CompletionResult, ModelInfo, ModelProvider
from chess_fight.providers.registry import register_provider

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

    def __init__(self):
        self.base_url = os.getenv(
            "GROQ_BASE_URL",
            "https://api.groq.com/openai/v1",
        )

    def validate_key(self, api_key: str) -> bool:
        return api_key.startswith("gsk_") and len(api_key) > 20

    async def list_models(self, api_key: str) -> list[ModelInfo]:
        client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)
        models = await client.models.list()

        chat_models = []
        seen = set()
        for model in models.data:
            model_id = model.id
            if model_id in seen:
                continue
            seen.add(model_id)

            # Include known free models and any chat-capable models
            if any(
                prefix in model_id
                for prefix in [
                    "llama",
                    "mixtral",
                    "gemma",
                    "deepseek",
                    "qwen",
                    "mistral",
                    "command",
                ]
            ):
                is_free = model_id in _GROQ_FREE_MODELS
                chat_models.append(
                    ModelInfo(
                        id=model_id,
                        name=model_id + (" ★free" if is_free else ""),
                        provider="groq",
                        context_window=8192,
                        pricing_tier="free" if is_free else "paid",
                    )
                )
        return chat_models

    async def complete(
        self,
        api_key: str,
        model: str,
        messages: list[ChatMessage],
        **params,
    ) -> CompletionResult:
        client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)

        openai_messages: list[ChatCompletionMessageParam] = [
            {"role": m.role, "content": m.content}  # type: ignore[misc]
            for m in messages
        ]

        temperature = params.get("temperature", 0.1)
        max_tokens = params.get("max_tokens", 100)

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
            tokens_out=response.usage.completion_tokens
            if response.usage
            else None,
            latency_ms=latency_ms,
            raw_response=response.model_dump()
            if hasattr(response, "model_dump")
            else None,
        )
