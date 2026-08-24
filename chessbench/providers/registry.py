"""Provider registry for managing model providers."""


import importlib
import logging

from chessbench.common.common_types import ModelProvider

_log = logging.getLogger(__name__)

PROVIDER_REGISTRY: dict[str, type[ModelProvider]] = {}

# Modules whose import side effect registers a provider. Kept here so both
# the package __getattr__ and the auto-registration below share one list.
LAZY_PROVIDER_MODULES: tuple[str, ...] = (
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

_modules_loaded = False


def ensure_providers_registered() -> None:
    """Import every provider module once so ``@register_provider`` decorators run.

    Idempotent. Import happens lazily at call time (not module import time) to
    keep Streamlit Cloud startup cheap; without this, ``get_provider`` fails
    with "Unknown provider" whenever callers skip the package-level
    ``list_providers()`` warm-up.
    """
    global _modules_loaded
    if _modules_loaded:
        return
    _modules_loaded = True
    for name in LAZY_PROVIDER_MODULES:
        try:
            importlib.import_module(f"chessbench.providers.{name}")
        except ImportError as exc:
            _log.warning("Provider module '%s' failed to import: %s", name, exc)


def register_provider(cls: type[ModelProvider]) -> type[ModelProvider]:
    """Register a provider class."""
    PROVIDER_REGISTRY[cls.name] = cls
    return cls


def get_provider(name: str) -> ModelProvider | None:
    """Get a provider instance by name."""
    ensure_providers_registered()
    provider_cls = PROVIDER_REGISTRY.get(name)
    if provider_cls:
        return provider_cls()
    return None


def list_providers() -> list[str]:
    """List all registered provider names."""
    ensure_providers_registered()
    return sorted(PROVIDER_REGISTRY.keys())
