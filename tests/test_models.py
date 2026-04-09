"""Tests for the models module.

Tests model configuration, registry, and cost calculation utilities.
"""

import pytest

from tui_chatbot.models import (
    KNOWN_MODELS,
    Model,
    calculate_cost,
    format_cost,
    get_model,
    get_model_or_default,
    get_models_by_provider,
    list_models,
    register_model,
)
from tui_chatbot.types import Usage


# ╭────────────────────────────────────────────────────────────╮
# │  Model Dataclass Tests                                       │
# ╰────────────────────────────────────────────────────────────╯


def test_model_creation():
    """Test creating a Model dataclass."""
    model = Model(
        id="test-model",
        name="Test Model",
        api="chat.completions",
        provider="openai",
        base_url="https://api.example.com",
        reasoning=True,
        input_cost=1.0,
        output_cost=2.0,
        context_window=4096,
        max_tokens=1024,
    )

    assert model.id == "test-model"
    assert model.name == "Test Model"
    assert model.api == "chat.completions"
    assert model.provider == "openai"
    assert model.base_url == "https://api.example.com"
    assert model.reasoning is True
    assert model.input_cost == 1.0
    assert model.output_cost == 2.0
    assert model.context_window == 4096
    assert model.max_tokens == 1024


def test_model_defaults():
    """Test Model dataclass default values."""
    model = Model(id="test", name="Test")

    assert model.api == "chat.completions"
    assert model.provider == "openai"
    assert model.base_url is None
    assert model.reasoning is False
    assert model.input_cost == 0.0
    assert model.output_cost == 0.0
    assert model.context_window == 8192
    assert model.max_tokens is None


# ╭────────────────────────────────────────────────────────────╮
# │  Known Models Tests                                          │
# ╰────────────────────────────────────────────────────────────╯


def test_known_models_contains_openai():
    """Test that OpenAI models are in KNOWN_MODELS."""
    assert "gpt-4" in KNOWN_MODELS
    assert "gpt-4o" in KNOWN_MODELS
    assert "gpt-3.5-turbo" in KNOWN_MODELS
    assert "o1" in KNOWN_MODELS
    assert "o1-mini" in KNOWN_MODELS


def test_known_models_contains_doubao():
    """Test that Doubao models are in KNOWN_MODELS."""
    assert "doubao-seed-1.5" in KNOWN_MODELS
    assert "doubao-pro" in KNOWN_MODELS
    assert "doubao-lite" in KNOWN_MODELS


def test_openai_model_config():
    """Test OpenAI model configuration."""
    gpt4 = KNOWN_MODELS["gpt-4"]

    assert gpt4.name == "GPT-4"
    assert gpt4.provider == "openai"
    assert gpt4.input_cost == 30.0
    assert gpt4.output_cost == 60.0
    assert gpt4.context_window == 8192
    assert gpt4.reasoning is False


def test_o1_model_has_reasoning():
    """Test that o1 models have reasoning enabled."""
    o1 = KNOWN_MODELS["o1"]
    assert o1.reasoning is True

    o1_mini = KNOWN_MODELS["o1-mini"]
    assert o1_mini.reasoning is True

    o3_mini = KNOWN_MODELS["o3-mini"]
    assert o3_mini.reasoning is True


def test_doubao_model_config():
    """Test Doubao model configuration."""
    doubao = KNOWN_MODELS["doubao-seed-1.5"]

    assert doubao.name == "Doubao Seed 1.5"
    assert doubao.provider == "doubao"
    assert doubao.base_url == "https://ark.cn-beijing.volces.com/api/v3"
    assert doubao.reasoning is True


# ╭────────────────────────────────────────────────────────────╮
# │  Model Retrieval Tests                                       │
# ╰────────────────────────────────────────────────────────────╯


def test_get_model_success():
    """Test getting a known model."""
    model = get_model("gpt-4")

    assert model.id == "gpt-4"
    assert model.name == "GPT-4"


def test_get_model_not_found():
    """Test getting an unknown model raises KeyError."""
    with pytest.raises(KeyError, match="Unknown model"):
        get_model("unknown-model-xyz")


def test_get_model_or_default_known():
    """Test get_model_or_default returns known model."""
    model = get_model_or_default("gpt-4")

    assert model.id == "gpt-4"
    assert model.name == "GPT-4"


def test_get_model_or_default_unknown():
    """Test get_model_or_default returns default for unknown."""
    model = get_model_or_default("custom-model-123")

    assert model.id == "custom-model-123"
    assert model.name == "custom-model-123"
    assert model.api == "chat.completions"
    assert model.provider == "openai"


def test_list_models():
    """Test listing all model IDs."""
    models = list_models()

    assert "gpt-4" in models
    assert "gpt-4o" in models
    assert "doubao-seed-1.5" in models


def test_get_models_by_provider():
    """Test getting models by provider."""
    openai_models = get_models_by_provider("openai")
    doubao_models = get_models_by_provider("doubao")

    openai_ids = [m.id for m in openai_models]
    doubao_ids = [m.id for m in doubao_models]

    assert "gpt-4" in openai_ids
    assert "gpt-4o" in openai_ids
    assert "doubao-seed-1.5" in doubao_ids
    assert "doubao-pro" in doubao_ids

    # No overlap
    assert not set(openai_ids) & set(doubao_ids)


# ╭────────────────────────────────────────────────────────────╮
# │  Model Registration Tests                                    │
# ╰────────────────────────────────────────────────────────────╯


def test_register_model():
    """Test registering a new model."""
    model = Model(
        id="custom-model",
        name="Custom Model",
        provider="openai",
        input_cost=0.5,
    )

    register_model(model)

    assert "custom-model" in KNOWN_MODELS
    assert KNOWN_MODELS["custom-model"].name == "Custom Model"

    # Cleanup
    del KNOWN_MODELS["custom-model"]


# ╭────────────────────────────────────────────────────────────╮
# │  Cost Calculation Tests                                      │
# ╰────────────────────────────────────────────────────────────╯


def test_calculate_cost_basic():
    """Test basic cost calculation."""
    model = Model(
        id="test",
        name="Test",
        input_cost=1.0,  # $1 per 1M tokens
        output_cost=2.0,  # $2 per 1M tokens
    )

    usage = Usage(inputTokens=1_000_000, outputTokens=500_000)
    cost = calculate_cost(usage, model)

    # $1 for input + $1 for output (half of $2) = $2
    assert cost == 2.0


def test_calculate_cost_zero():
    """Test cost calculation with zero usage."""
    model = Model(
        id="test",
        name="Test",
        input_cost=1.0,
        output_cost=2.0,
    )

    usage = Usage(inputTokens=0, outputTokens=0)
    cost = calculate_cost(usage, model)

    assert cost == 0.0


def test_calculate_cost_gpt4():
    """Test cost calculation for GPT-4."""
    gpt4 = KNOWN_MODELS["gpt-4"]

    usage = Usage(inputTokens=1000, outputTokens=500)
    cost = calculate_cost(usage, gpt4)

    # GPT-4: $30 per 1M input, $60 per 1M output
    # 1000 input tokens = $0.03
    # 500 output tokens = $0.03
    # Total = $0.06
    expected = (1000 / 1_000_000) * 30.0 + (500 / 1_000_000) * 60.0
    assert abs(cost - expected) < 0.0001


def test_calculate_cost_with_cache():
    """Test cost calculation with cache information."""
    model = Model(
        id="test",
        name="Test",
        input_cost=1.0,
        output_cost=2.0,
    )

    usage = Usage(
        inputTokens=1_000_000,
        outputTokens=500_000,
        cacheRead=100_000,
        cacheWrite=50_000,
    )
    cost = calculate_cost(usage, model)

    # Should only count input/output tokens, not cache
    assert cost == 2.0


def test_format_cost_small():
    """Test formatting small costs."""
    assert format_cost(0.000001) == "$0.000001"
    assert format_cost(0.00001) == "$0.000010"


def test_format_cost_medium():
    """Test formatting medium costs."""
    assert format_cost(0.01) == "$0.0100"
    assert format_cost(0.5) == "$0.5000"
    assert format_cost(0.99) == "$0.9900"


def test_format_cost_large():
    """Test formatting large costs."""
    assert format_cost(1.0) == "$1.00"
    assert format_cost(100.50) == "$100.50"


def test_calculate_cost_rounding():
    """Test that cost is rounded to 6 decimal places."""
    model = Model(
        id="test",
        name="Test",
        input_cost=0.1234567,
        output_cost=0.0,
    )

    usage = Usage(inputTokens=1_000_000, outputTokens=0)
    cost = calculate_cost(usage, model)

    # Should be rounded to 6 decimal places
    assert len(str(cost).split(".")[-1]) <= 6
