"""Ollama provider implementation (local models)."""

import os

import httpx

from .base import ChatMessage, CompletionResult, ModelInfo, ModelProvider
from .registry import register_provider


@register_provider
class OllamaProvider(ModelProvider):
    name = "ollama"
    requires_api_key = False

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def validate_key(self, api_key: str) -> bool:
        return True  # No API key needed

    async def list_models(self, api_key: str = "") -> list[ModelInfo]:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()

            chat_models = []
            for model in data.get("models", []):
                model_id = model["name"]
                chat_models.append(ModelInfo(
                    id=model_id,
                    name=model_id,
                    provider="ollama",
                    context_window=8192,  # Varies by model
                ))
            return chat_models

    async def complete(self, api_key: str, model: str, messages: list[ChatMessage], **params) -> CompletionResult:
        async with httpx.AsyncClient() as client:
            # Convert messages to Ollama format (single prompt)
            prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)

            temperature = params.get("temperature", 0.1)
            max_tokens = params.get("max_tokens", 100)

            import time
            start = time.time()

            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                },
                timeout=60.0
            )
            response.raise_for_status()

            data = response.json()
            latency_ms = int((time.time() - start) * 1000)

            return CompletionResult(
                text=data.get("response", ""),
                tokens_in=data.get("prompt_eval_count"),
                tokens_out=data.get("eval_count"),
                latency_ms=latency_ms,
                raw_response=data
            )
