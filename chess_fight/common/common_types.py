"""Common types shared between models and providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a model from a provider."""
    id: str
    name: str
    provider: str
    context_window: int | None = None
    pricing_tier: str | None = None
    capabilities: list[str] | None = None


@dataclass
class ChatMessage:
    """Chat message for completion."""
    role: str
    content: str


@dataclass
class CompletionResult:
    """Result from a completion request."""
    text: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: int | None = None
    raw_response: dict | None = None


class ModelProvider(ABC):
    """Abstract base class for model providers."""

    name: str
    requires_api_key: bool = True

    @abstractmethod
    async def list_models(self, api_key: str) -> list["ModelInfo"]:
        """List available models from the provider."""

    @abstractmethod
    async def complete(self, api_key: str, model: str, messages: list["ChatMessage"], **params) -> "CompletionResult":
        """Complete a chat conversation."""

    @abstractmethod
    def validate_key(self, api_key: str) -> bool:
        """Validate an API key format (doesn't make network calls)."""
