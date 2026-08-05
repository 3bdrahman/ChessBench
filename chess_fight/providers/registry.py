"""Provider registry for managing model providers."""


from .base import ModelProvider

PROVIDER_REGISTRY: dict[str, type[ModelProvider]] = {}


def register_provider(cls: type[ModelProvider]) -> type[ModelProvider]:
    """Register a provider class."""
    PROVIDER_REGISTRY[cls.name] = cls
    return cls


def get_provider(name: str) -> ModelProvider | None:
    """Get a provider instance by name."""
    provider_cls = PROVIDER_REGISTRY.get(name)
    if provider_cls:
        return provider_cls()
    return None


def list_providers() -> list[str]:
    """List all registered provider names."""
    return list(PROVIDER_REGISTRY.keys())
