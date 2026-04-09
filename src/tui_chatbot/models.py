"""Model configuration and registry.

Type-safe model definitions with cost calculation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .types import Usage


# ╭────────────────────────────────────────────────────────────╮
# │  Model Dataclass                                             │
# ╰────────────────────────────────────────────────────────────╯


@dataclass
class Model:
    """Model configuration with metadata and pricing.

    Attributes:
        id: Model identifier (e.g., "gpt-4", "doubao-seed-1.5")
        name: Human-readable name (e.g., "GPT-4")
        api: API endpoint type (e.g., "chat.completions")
        provider: Provider registry key (e.g., "openai", "doubao")
        base_url: Optional custom base URL for the API
        reasoning: Whether this model supports reasoning content
        input_cost: Cost per 1M input tokens (in USD)
        output_cost: Cost per 1M output tokens (in USD)
        context_window: Maximum context window size in tokens
        max_tokens: Maximum tokens to generate (None for model default)
    """

    id: str
    name: str
    api: str = "chat.completions"
    provider: str = "openai"
    base_url: Optional[str] = None
    reasoning: bool = False
    input_cost: float = 0.0  # per 1M tokens
    output_cost: float = 0.0  # per 1M tokens
    context_window: int = 8192
    max_tokens: Optional[int] = None


# ╭────────────────────────────────────────────────────────────╮
# │  Known Models Registry                                       │
# ╰────────────────────────────────────────────────────────────╯


KNOWN_MODELS: dict[str, Model] = {
    # OpenAI models
    "gpt-4": Model(
        id="gpt-4",
        name="GPT-4",
        api="chat.completions",
        provider="openai",
        input_cost=30.0,
        output_cost=60.0,
        context_window=8192,
    ),
    "gpt-4-turbo": Model(
        id="gpt-4-turbo",
        name="GPT-4 Turbo",
        api="chat.completions",
        provider="openai",
        input_cost=10.0,
        output_cost=30.0,
        context_window=128000,
    ),
    "gpt-4o": Model(
        id="gpt-4o",
        name="GPT-4o",
        api="chat.completions",
        provider="openai",
        input_cost=5.0,
        output_cost=15.0,
        context_window=128000,
    ),
    "gpt-4o-mini": Model(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        api="chat.completions",
        provider="openai",
        input_cost=0.15,
        output_cost=0.60,
        context_window=128000,
    ),
    "gpt-3.5-turbo": Model(
        id="gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        api="chat.completions",
        provider="openai",
        input_cost=0.50,
        output_cost=1.50,
        context_window=16385,
    ),
    "o1": Model(
        id="o1",
        name="o1",
        api="chat.completions",
        provider="openai",
        reasoning=True,
        input_cost=15.0,
        output_cost=60.0,
        context_window=128000,
    ),
    "o1-mini": Model(
        id="o1-mini",
        name="o1-mini",
        api="chat.completions",
        provider="openai",
        reasoning=True,
        input_cost=3.0,
        output_cost=12.0,
        context_window=128000,
    ),
    "o3-mini": Model(
        id="o3-mini",
        name="o3-mini",
        api="chat.completions",
        provider="openai",
        reasoning=True,
        input_cost=1.10,
        output_cost=4.40,
        context_window=200000,
    ),
    # Doubao / Ark / Volces models (using OpenAI-compatible API)
    "doubao-seed-1.5": Model(
        id="doubao-seed-1.5",
        name="Doubao Seed 1.5",
        api="chat.completions",
        provider="doubao",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        reasoning=True,
        input_cost=2.0,
        output_cost=8.0,
        context_window=32000,
    ),
    "doubao-pro": Model(
        id="doubao-pro",
        name="Doubao Pro",
        api="chat.completions",
        provider="doubao",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        input_cost=0.8,
        output_cost=2.0,
        context_window=32000,
    ),
    "doubao-lite": Model(
        id="doubao-lite",
        name="Doubao Lite",
        api="chat.completions",
        provider="doubao",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        input_cost=0.3,
        output_cost=0.6,
        context_window=32000,
    ),
}


# ╭────────────────────────────────────────────────────────────╮
# │  Model Resolution Utilities                                  │
# ╰────────────────────────────────────────────────────────────╯


def get_model(model_id: str) -> Model:
    """Get model configuration by ID.

    Args:
        model_id: Model identifier

    Returns:
        Model configuration

    Raises:
        KeyError: If model is not found in registry
    """
    if model_id not in KNOWN_MODELS:
        raise KeyError(f"Unknown model: {model_id}")
    return KNOWN_MODELS[model_id]


def get_model_or_default(model_id: str) -> Model:
    """Get model configuration or return a default config.

    Args:
        model_id: Model identifier

    Returns:
        Model configuration (returns generic config if not found)
    """
    if model_id in KNOWN_MODELS:
        return KNOWN_MODELS[model_id]
    # Return a generic model config
    return Model(
        id=model_id,
        name=model_id,
        api="chat.completions",
        provider="openai",
    )


def register_model(model: Model) -> None:
    """Register a new model in the registry.

    Args:
        model: Model configuration to register
    """
    KNOWN_MODELS[model.id] = model


def list_models() -> list[str]:
    """List all registered model IDs.

    Returns:
        List of model identifiers
    """
    return list(KNOWN_MODELS.keys())


def get_models_by_provider(provider: str) -> list[Model]:
    """Get all models for a specific provider.

    Args:
        provider: Provider registry key

    Returns:
        List of model configurations
    """
    return [m for m in KNOWN_MODELS.values() if m.provider == provider]


# ╭────────────────────────────────────────────────────────────╮
# │  Cost Calculation                                            │
# ╰────────────────────────────────────────────────────────────╯


def calculate_cost(usage: Usage, model: Model) -> float:
    """Calculate cost based on token usage and model pricing.

    Args:
        usage: Token usage statistics
        model: Model configuration with pricing

    Returns:
        Total cost in USD
    """
    if not usage:
        return 0.0

    # Costs are per 1M tokens
    input_cost = (usage.inputTokens / 1_000_000) * model.input_cost
    output_cost = (usage.outputTokens / 1_000_000) * model.output_cost

    return round(input_cost + output_cost, 6)


def format_cost(cost: float) -> str:
    """Format cost for display.

    Args:
        cost: Cost in USD

    Returns:
        Formatted cost string
    """
    if cost < 0.0001:
        return f"${cost:.6f}"
    elif cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1.0:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"
