"""Provider abstraction layer for LLM chess AI."""

# Import concrete providers to register them
from . import (
    anthropic,  # noqa: F401
    google,  # noqa: F401
    groq,  # noqa: F401
    nim,  # noqa: F401
    ollama,  # noqa: F401
    openai,  # noqa: F401
    openrouter,  # noqa: F401
)
from .base import ChatMessage, CompletionResult, ModelInfo, ModelProvider
from .chess_ai import ProviderChessAI
from .registry import PROVIDER_REGISTRY, get_provider, list_providers, register_provider

__all__ = [
    "PROVIDER_REGISTRY",
    "ChatMessage",
    "CompletionResult",
    "ModelInfo",
    "ModelProvider",
    "ProviderChessAI",
    "get_provider",
    "list_providers",
    "register_provider",
]
