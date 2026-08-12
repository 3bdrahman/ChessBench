"""Ollama provider implementation (local models)."""

import logging
import os
import time
from typing import Any

import httpx

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
    ConnectionError,
    ModelNotFoundError,
    ProviderAPIError,
    ProviderUnavailableError,
    RateLimitError,
    TimeoutError,
)
from chess_fight.providers.registry import register_provider

_log = logging.getLogger(__name__)


@register_provider
class OllamaProvider(ModelProvider):
    name = "ollama"
    requires_api_key = False

    def __init__(self) -> None:
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def validate_key(self, api_key: str) -> bool:
        return True  # No API key needed

    async def list_models(self, api_key: str = "") -> list[ModelInfo]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            _log.warning("Ollama list_models connection error: %s", exc)
            raise ProviderUnavailableError("ollama", f"Cannot connect to {self.base_url}") from exc
        except httpx.TimeoutException as exc:
            _log.warning("Ollama list_models timeout: %s", exc)
            raise TimeoutError("ollama", 10.0, self.base_url) from exc
        except httpx.HTTPError as exc:
            _log.warning("Ollama list_models HTTP error: %s", exc)
            return []
        except Exception as exc:
            _log.warning("Ollama list_models error: %s", exc)
            return []

        chat_models: list[ModelInfo] = []
        for model in data.get("models", []):
            model_id = model["name"]
            mid = model_id.lower()
            # Drop pure-embedding Ollama variants (nomic-embed, mxbai-embed, all-minilm).
            if any(tok in mid for tok in ("embed", "bge-", "e5-", "gte-")):
                continue
            chat_models.append(ModelInfo(
                id=model_id,
                name=model_id,
                provider="ollama",
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
        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
        temperature = params.get("temperature", DEFAULT_TEMPERATURE)
        max_tokens = params.get("max_tokens")

        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": options,
                    },
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            latency_ms = int((time.time() - start) * 1000)
            raise TimeoutError("ollama", 60.0, self.base_url) from exc
        except httpx.ConnectError as exc:
            latency_ms = int((time.time() - start) * 1000)
            raise ProviderUnavailableError("ollama", f"Cannot connect to {self.base_url}") from exc
        except httpx.HTTPStatusError as exc:
            latency_ms = int((time.time() - start) * 1000)
            status = exc.response.status_code
            if status == 404:
                raise ModelNotFoundError("ollama", model, available_models=None) from exc
            if status == 429:
                raise RateLimitError("ollama", retry_after=None, http_status=429) from exc
            if 500 <= status < 600:
                raise ProviderAPIError("ollama", status, str(exc), raw_response={"error": str(exc)}) from exc
            raise ProviderAPIError("ollama", status, str(exc), raw_response={"error": str(exc)}) from exc
        except httpx.HTTPError as exc:
            latency_ms = int((time.time() - start) * 1000)
            raise ConnectionError("ollama", self.base_url, str(exc)) from exc
        except Exception as exc:
            latency_ms = int((time.time() - start) * 1000)
            _log.warning("Ollama complete error: %s", exc)
            raise ProviderAPIError("ollama", 500, str(exc), raw_response={"error": str(exc)}) from exc

        latency_ms = int((time.time() - start) * 1000)

        return CompletionResult(
            text=data.get("response", ""),
            tokens_in=data.get("prompt_eval_count"),
            tokens_out=data.get("eval_count"),
            latency_ms=latency_ms,
            raw_response=data,
        )
