"""Provider registry with lazy loading support.

Global registry for managing LLM providers with support for:
- Registration with factory functions (lazy loading)
- Provider aliases
- Model-based provider resolution
"""

from __future__ import annotations

import asyncio
from typing import Callable, Optional, TypeVar

from .base import Provider

# Type alias for provider factory functions
ProviderFactory = Callable[[], Provider]

# Registry storage
_registry: dict[str, ProviderFactory] = {}
_aliases: dict[str, str] = {}
_cache: dict[str, Provider] = {}

T = TypeVar("T", bound=Provider)


# ╭────────────────────────────────────────────────────────────╮
# │  Registration                                                │
# ╰────────────────────────────────────────────────────────────╯


def register(
    name: str,
    factory: ProviderFactory,
    aliases: Optional[list[str]] = None,
) -> None:
    """Register a provider with the registry.

    Args:
        name: Primary provider identifier (e.g., "openai", "doubao")
        factory: Factory function that returns a Provider instance
        aliases: Optional list of aliases for this provider

    Example:
        register("openai", lambda: OpenAIProvider())
        register(
            "doubao",
            lambda: OpenAIProvider(base_url="https://..."),
            aliases=["seed", "ark"]
        )
    """
    _registry[name] = factory

    # Register aliases
    if aliases:
        for alias in aliases:
            _aliases[alias] = name


def register_lazy(
    name: str,
    factory: Callable[[], Provider],
    aliases: Optional[list[str]] = None,
) -> None:
    """Register a provider with lazy loading (alias for register).

    This is an alias for register() - all registrations use factory functions
    for lazy loading by default.

    Args:
        name: Primary provider identifier
        factory: Factory function returning Provider instance
        aliases: Optional list of aliases
    """
    register(name, factory, aliases)


# ╭────────────────────────────────────────────────────────────╮
# │  Retrieval                                                   │
# ╰────────────────────────────────────────────────────────────╯


def _resolve_name(name: str) -> str:
    """Resolve a name or alias to the primary provider name.

    Args:
        name: Provider name or alias

    Returns:
        Resolved primary provider name
    """
    # Check if it's a direct registration
    if name in _registry:
        return name

    # Check aliases
    if name in _aliases:
        return _aliases[name]

    return name


def get(name: str) -> Provider:
    """Get a provider instance by name.

    Uses lazy loading - the provider is created on first access
    and cached for subsequent calls.

    Args:
        name: Provider name or alias

    Returns:
        Provider instance

    Raises:
        KeyError: If provider is not registered
    """
    resolved = _resolve_name(name)

    # Return cached instance if available
    if resolved in _cache:
        return _cache[resolved]

    # Create and cache new instance
    if resolved not in _registry:
        raise KeyError(f"Provider not registered: {name} (resolved: {resolved})")

    factory = _registry[resolved]
    provider = factory()

    if not isinstance(provider, Provider):
        raise TypeError(
            f"Factory for '{resolved}' must return a Provider instance, "
            f"got {type(provider).__name__}"
        )

    _cache[resolved] = provider
    return provider


def get_for_model(model_id: str) -> Provider:
    """Get the appropriate provider for a model.

    Resolves the provider based on model ID prefix or exact match
    using the model registry.

    Args:
        model_id: Model identifier (e.g., "gpt-4", "doubao-seed-1.5")

    Returns:
        Provider instance for the model

    Raises:
        KeyError: If no provider is found for the model
    """
    from ..models import get_model_or_default

    model = get_model_or_default(model_id)
    return get(model.provider)


def list_providers() -> list[str]:
    """List all registered provider names.

    Returns:
        List of registered provider identifiers (not aliases)
    """
    return list(_registry.keys())


def list_aliases() -> dict[str, str]:
    """List all registered aliases.

    Returns:
        Dictionary mapping alias -> primary provider name
    """
    return dict(_aliases)


def clear() -> None:
    """Clear the registry and cache.

    Mainly useful for testing to ensure a clean state.
    """
    _registry.clear()
    _aliases.clear()
    _cache.clear()


# ╭────────────────────────────────────────────────────────────╮
# │  Provider Creation Utilities                                 │
# ╰────────────────────────────────────────────────────────────╯


def create_provider_from_env(provider: str = "openai") -> Optional[Provider]:
    """Create a provider from environment variables.

    Args:
        provider: Provider type to create ("openai", "doubao")

    Returns:
        Provider instance if environment is configured, None otherwise

    Example:
        # With OPENAI_API_KEY set:
        provider = create_provider_from_env("openai")
        # Returns OpenAIProvider with API key from env
    """
    import os

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return None

        from .openai import OpenAIProvider, OpenAIProviderConfig

        config = OpenAIProviderConfig(api_key=api_key)
        return OpenAIProvider(config=config)

    if provider == "doubao":
        api_key = os.getenv("ARK_API_KEY", "") or os.getenv("DOUBAO_API_KEY", "")
        if not api_key:
            return None

        from .openai import OpenAIProvider, OpenAIProviderConfig

        config = OpenAIProviderConfig(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
        return OpenAIProvider(config=config)

    return None


def register_default_providers() -> None:
    """Register default providers from environment variables.

    Automatically detects available providers based on environment
    variables and registers them.
    """
    import os

    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        register(
            "openai",
            lambda: create_provider_from_env("openai"),
        )

    # Doubao / Ark / Volces
    if os.getenv("ARK_API_KEY") or os.getenv("DOUBAO_API_KEY"):
        register(
            "doubao",
            lambda: create_provider_from_env("doubao"),
            aliases=["ark", "volces"],
        )
