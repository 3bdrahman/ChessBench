"""Provider abstraction layer for LLM chess AI."""

from .base import ModelProvider, ModelInfo, CompletionResult, ChatMessage
from .registry import PROVIDER_REGISTRY, register_provider, get_provider, list_providers
from .chess_ai import ProviderChessAI

# Import concrete providers to register them
from . import openai  # noqa: F401
from . import anthropic  # noqa: F401
from . import google  # noqa: F401
from . import nim  # noqa: F401
from . import openrouter  # noqa: F401
from . import groq  # noqa: F401
from . import ollama  # noqa: F401

__all__ = [
    "ModelProvider",
    "ModelInfo", 
    "CompletionResult",
    "ChatMessage",
    "PROVIDER_REGISTRY",
    "register_provider",
    "get_provider",
    "list_providers",
    "ProviderChessAI",
]