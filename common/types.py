"""Common types shared between models and providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class ModelInfo:
    """Information about a model from a provider."""
    id: str
    name: str
    provider: str
    context_window: Optional[int] = None
    pricing_tier: Optional[str] = None
    capabilities: Optional[list[str]] = None


@dataclass
class ChatMessage:
    """Chat message for completion."""
    role: str
    content: str


@dataclass
class CompletionResult:
    """Result from a completion request."""
    text: str
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    latency_ms: Optional[int] = None
    raw_response: Optional[dict] = None


class ModelProvider(ABC):
    """Abstract base class for model providers."""

    name: str
    requires_api_key: bool = True

    @abstractmethod
    async def list_models(self, api_key: str) -> list["ModelInfo"]:
        """List available models from the provider."""
        pass

    @abstractmethod
    async def complete(self, api_key: str, model: str, messages: list["ChatMessage"], **params) -> "CompletionResult":
        """Complete a chat conversation."""
        pass

    @abstractmethod
    def validate_key(self, api_key: str) -> bool:
        """Validate an API key format (doesn't make network calls)."""
        pass