"""Provider abstraction layer for LLM chess AI."""

from chessbench.common.common_types import ChatMessage, CompletionResult, ModelInfo, ModelProvider
from chessbench.providers.registry import PROVIDER_REGISTRY, get_provider, register_provider

# Lazy imports - provider modules are imported on first access via get_provider()
# This avoids import-time circular dependencies and issues on Streamlit Cloud
_PROVIDER_MODULES = (
    "anthropic",
    "deepinfra",
    "fireworks",
    "generic_openai",
    "google",
    "groq",
    "nim",
    "openai",
    "openrouter",
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

from .chess_ai import ProviderChessAI


def list_providers() -> list[str]:
    """List all available providers, forcing lazy loading."""
    import sys
    this_module = sys.modules[__name__]
    for name in _PROVIDER_MODULES:
        getattr(this_module, name)
    from chessbench.providers.registry import list_providers as _registry_list
    return _registry_list()

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
