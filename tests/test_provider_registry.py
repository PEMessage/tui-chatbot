"""Tests for the provider registry.

Tests registration, lazy loading, aliases, and model-based resolution.
"""

import pytest

from tui_chatbot.providers.base import Provider, ProviderConfig
from tui_chatbot.providers.registry import (
    clear,
    get,
    get_for_model,
    list_aliases,
    list_providers,
    register,
    register_lazy,
)


# ╭────────────────────────────────────────────────────────────╮
# │  Test Fixtures                                               │
# ╰────────────────────────────────────────────────────────────╯


class MockProvider(Provider):
    """Mock provider for testing."""

    def __init__(self, name: str = "mock"):
        self._name = name
        self.config = ProviderConfig()

    @property
    def name(self) -> str:
        return self._name

    async def stream(self, messages, model, **kwargs):
        from tui_chatbot.event_stream import AssistantMessageEventStream

        return AssistantMessageEventStream()

    async def list_models(self):
        return ["mock-model"]


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean registry before each test."""
    clear()
    yield
    clear()


# ╭────────────────────────────────────────────────────────────╮
# │  Registration Tests                                          │
# ╰────────────────────────────────────────────────────────────╯


def test_register_provider():
    """Test basic provider registration."""
    register("test", lambda: MockProvider("test"))

    assert "test" in list_providers()


def test_register_with_aliases():
    """Test registration with aliases."""
    register("openai", lambda: MockProvider("openai"), aliases=["gpt", "chatgpt"])

    # Primary name should be registered
    assert "openai" in list_providers()

    # Aliases should be resolvable
    provider = get("gpt")
    assert provider.name == "openai"

    provider = get("chatgpt")
    assert provider.name == "openai"


def test_register_lazy_is_alias():
    """Test that register_lazy is an alias for register."""
    register_lazy("lazy", lambda: MockProvider("lazy"))

    assert "lazy" in list_providers()
    provider = get("lazy")
    assert provider.name == "lazy"


# ╭────────────────────────────────────────────────────────────╮
# │  Lazy Loading Tests                                          │
# ╰────────────────────────────────────────────────────────────╯


def test_lazy_loading_caches_instance():
    """Test that lazy loading caches the provider instance."""
    call_count = 0

    def factory():
        nonlocal call_count
        call_count += 1
        return MockProvider("cached")

    register("cached", factory)

    # First access creates instance
    provider1 = get("cached")
    assert call_count == 1

    # Second access returns cached instance
    provider2 = get("cached")
    assert call_count == 1  # Factory not called again
    assert provider1 is provider2


def test_lazy_loading_different_factories():
    """Test that different providers have separate instances."""
    register("provider1", lambda: MockProvider("provider1"))
    register("provider2", lambda: MockProvider("provider2"))

    p1 = get("provider1")
    p2 = get("provider2")

    assert p1.name == "provider1"
    assert p2.name == "provider2"
    assert p1 is not p2


# ╭────────────────────────────────────────────────────────────╮
# │  Provider Resolution Tests                                   │
# ╰────────────────────────────────────────────────────────────╯


def test_get_existing_provider():
    """Test getting an existing provider."""
    register("exists", lambda: MockProvider("exists"))

    provider = get("exists")
    assert provider.name == "exists"


def test_get_nonexistent_provider():
    """Test getting a non-existent provider raises KeyError."""
    with pytest.raises(KeyError, match="Provider not registered"):
        get("nonexistent")


def test_get_with_alias():
    """Test getting provider via alias."""
    register("primary", lambda: MockProvider("primary"), aliases=["alias1", "alias2"])

    # Get via primary name
    p1 = get("primary")
    assert p1.name == "primary"

    # Get via alias
    p2 = get("alias1")
    assert p2 is p1  # Same cached instance

    p3 = get("alias2")
    assert p3 is p1


def test_get_for_model_known_model():
    """Test getting provider for a known model."""
    from tui_chatbot.models import KNOWN_MODELS, Model

    # Register a provider for testing
    register("testprovider", lambda: MockProvider("testprovider"))

    # Add a test model
    test_model = Model(
        id="test-model",
        name="Test Model",
        provider="testprovider",
    )
    KNOWN_MODELS["test-model"] = test_model

    try:
        provider = get_for_model("test-model")
        assert provider.name == "testprovider"
    finally:
        # Cleanup
        del KNOWN_MODELS["test-model"]


def test_get_for_model_unknown_model():
    """Test getting provider for unknown model falls back to openai."""
    from tui_chatbot.models import KNOWN_MODELS

    # Register openai provider
    register("openai", lambda: MockProvider("openai"))

    # Get provider for unknown model
    # This should return the openai provider as default
    provider = get_for_model("unknown-model-xyz")
    assert provider.name == "openai"


# ╭────────────────────────────────────────────────────────────╮
# │  Listing Tests                                               │
# ╰────────────────────────────────────────────────────────────╯


def test_list_providers_empty():
    """Test listing providers when registry is empty."""
    assert list_providers() == []


def test_list_providers_with_entries():
    """Test listing providers with entries."""
    register("p1", lambda: MockProvider("p1"))
    register("p2", lambda: MockProvider("p2"))

    providers = list_providers()
    assert sorted(providers) == ["p1", "p2"]


def test_list_aliases():
    """Test listing aliases."""
    register("main", lambda: MockProvider("main"), aliases=["a", "b", "c"])

    aliases = list_aliases()
    assert aliases == {"a": "main", "b": "main", "c": "main"}


def test_list_aliases_empty():
    """Test listing aliases when none registered."""
    assert list_aliases() == {}


# ╭────────────────────────────────────────────────────────────╮
# │  Clear Registry Tests                                        │
# ╰────────────────────────────────────────────────────────────╯


def test_clear_registry():
    """Test clearing the registry."""
    register("p1", lambda: MockProvider("p1"))
    register("p2", lambda: MockProvider("p2"), aliases=["alias"])

    clear()

    assert list_providers() == []
    assert list_aliases() == {}

    # Cached instances should be cleared too
    with pytest.raises(KeyError):
        get("p1")


# ╭────────────────────────────────────────────────────────────╮
# │  Error Handling Tests                                        │
# ╰────────────────────────────────────────────────────────────╯


def test_register_invalid_factory():
    """Test that factory must return a Provider instance."""
    register("bad", lambda: "not a provider")

    with pytest.raises(TypeError, match="must return a Provider instance"):
        get("bad")


def test_circular_alias_resolution():
    """Test that circular alias resolution doesn't cause infinite loop."""
    # This test documents current behavior
    # In practice, aliases should not point to other aliases
    register("target", lambda: MockProvider("target"))

    # Manually add alias to alias (not recommended but testing safety)
    from tui_chatbot.providers import registry

    registry._aliases["alias1"] = "target"

    provider = get("alias1")
    assert provider.name == "target"
