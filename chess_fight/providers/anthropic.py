"""Anthropic provider implementation."""

from typing import Literal, cast

from anthropic import AsyncAnthropic
from anthropic._types import NOT_GIVEN, NotGiven
from anthropic.types import MessageParam

from chess_fight.providers.base import ChatMessage, CompletionResult, ModelInfo, ModelProvider
from chess_fight.providers.registry import register_provider


@register_provider
class AnthropicProvider(ModelProvider):
    name = "anthropic"
    requires_api_key = True

    def validate_key(self, api_key: str) -> bool:
        return api_key.startswith("sk-ant-") and len(api_key) > 30

    async def list_models(self, api_key: str) -> list[ModelInfo]:
        known_models = [
            ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet"),
            ("claude-3-5-haiku-20241022", "Claude 3.5 Haiku"),
            ("claude-3-opus-20240229", "Claude 3 Opus"),
            ("claude-3-sonnet-20240229", "Claude 3 Sonnet"),
            ("claude-3-haiku-20240307", "Claude 3 Haiku"),
        ]
        return [
            ModelInfo(
                id=model_id,
                name=name,
                provider="anthropic",
                context_window=200000,
            )
            for model_id, name in known_models
        ]

    async def complete(self, api_key: str, model: str, messages: list[ChatMessage], **params) -> CompletionResult:
        client = AsyncAnthropic(api_key=api_key)

        system_prompt: str | NotGiven = NOT_GIVEN
        user_messages: list[MessageParam] = []
        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            else:
                role = cast(Literal["user", "assistant"], m.role)
                user_messages.append(MessageParam(role=role, content=m.content))

        temperature = params.get("temperature", 0.1)
        max_tokens = params.get("max_tokens", 100)

        import time
        start = time.time()

        response = await client.messages.create(
            model=model,
            system=system_prompt,  # type: ignore[arg-type]
            messages=user_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        latency_ms = int((time.time() - start) * 1000)

        text = ""
        if response.content:
            first_block = response.content[0]
            if hasattr(first_block, 'text'):
                text = first_block.text

        return CompletionResult(
            text=text,
            tokens_in=response.usage.input_tokens if response.usage else None,
            tokens_out=response.usage.output_tokens if response.usage else None,
            latency_ms=latency_ms,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
        )
