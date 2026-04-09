"""Configuration for TUI Chatbot.

Centralized configuration management for the chatbot.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Config:
    """Chatbot configuration.

    Attributes:
        base_url: API base URL
        api_key: API key for authentication
        model: Model identifier to use
        debug: Enable debug logging
        history: Maximum conversation history to keep
        reasoning_effort: Reasoning effort for o1/o3 models
    """

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    debug: bool = False
    history: int = 10
    reasoning_effort: Optional[str] = None

    def with_model(self, model: str) -> "Config":
        """Return a new config with a different model."""
        from dataclasses import replace

        return replace(self, model=model)
