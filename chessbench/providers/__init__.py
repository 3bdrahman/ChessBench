"""Provider abstraction layer for LLM chess AI."""

from chessbench.common.common_types import ChatMessage, CompletionResult, ModelInfo, ModelProvider
from chessbench.providers.registry import (
    PROVIDER_REGISTRY,
    ensure_providers_registered,
    get_provider,
    list_providers,
    register_provider,
)

# Lazy imports - provider modules are imported on first access via get_provider()
# This avoids import-time circular dependencies and issues on Streamlit Cloud
_PROVIDER_MODULES = (
    "anthropic",
    "deepinfra",
    "fireworks",
    "google",
    "groq",
    "nim",
    "ollama",
    "openai",
    "openrouter",
    "stockfish",
    "together",
)

def __getattr__(name: str):
    if name in _PROVIDER_MODULES:
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return list(globals().keys()) + list(_PROVIDER_MODULES)

from .chess_ai import ProviderChessAI  # noqa: E402  (deferred: chess_ai imports registry)

__all__ = [
    "PROVIDER_REGISTRY",
    "ChatMessage",
    "CompletionResult",
    "ModelInfo",
    "ModelProvider",
    "ProviderChessAI",
    "ensure_providers_registered",
    "get_provider",
    "list_providers",
    "register_provider",
]
