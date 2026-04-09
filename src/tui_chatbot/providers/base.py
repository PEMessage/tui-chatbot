"""Provider base interface.

Abstract base class defining the Provider protocol for LLM providers.
All providers must implement this interface for compatibility with the system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator, Optional

if TYPE_CHECKING:
    from ..event_stream import AssistantMessageEventStream
    from ..types import Message


# ╭────────────────────────────────────────────────────────────╮
# │  Provider Configuration                                      │
# ╰────────────────────────────────────────────────────────────╯


@dataclass
class ProviderConfig:
    """Base provider configuration.

    Attributes:
        api_key: API key for authentication
        base_url: Base URL for the API endpoint
        timeout: Request timeout in seconds
    """

    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    timeout: Optional[float] = 60.0


# ╭────────────────────────────────────────────────────────────╮
# │  Provider Interface                                          │
# ╰────────────────────────────────────────────────────────────╯


class Provider(ABC):
    """Abstract base class for LLM providers.

    All LLM providers must inherit from this class and implement
    the abstract methods to be compatible with the chat system.

    Example:
        class MyProvider(Provider):
            @property
            def name(self) -> str:
                return "myprovider"

            async def stream(
                self, messages, model, **kwargs
            ) -> AssistantMessageEventStream:
                # Implementation here
                pass
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'doubao')."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        model: str,
        **kwargs,
    ) -> AssistantMessageEventStream:
        """Stream chat completions with full event protocol.

        Args:
            messages: List of conversation messages
            model: Model identifier to use
            **kwargs: Additional provider-specific parameters
                (temperature, max_tokens, etc.)

        Returns:
            AssistantMessageEventStream emitting the full event protocol:
            - start
            - text_start -> text_delta* -> text_end
            - thinking_start -> thinking_delta* -> thinking_end (if reasoning)
            - toolcall_start -> toolcall_delta* -> toolcall_end (if tools)
            - done | error

        Example:
            stream = await provider.stream(messages, "gpt-4")
            async for event in stream:
                print(event.type, event.partial)
            result = await stream.result()
        """
        ...

    async def stream_simple(
        self,
        messages: list[Message],
        model: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream chat completions with simple text output.

                Convenience method that wraps stream() and yields only
        text deltas for simple use cases.

                Args:
                    messages: List of conversation messages
                    model: Model identifier to use
                    **kwargs: Additional provider-specific parameters

                Yields:
                    Text content chunks as they arrive

                Example:
                    async for text in provider.stream_simple(messages, "gpt-4"):
                        print(text, end="", flush=True)
        """
        from ..events import TextDeltaEvent

        stream = await self.stream(messages, model, **kwargs)
        async for event in stream:
            if isinstance(event, TextDeltaEvent):
                yield event.delta

    @abstractmethod
    async def list_models(self) -> list[str]:
        """List available models from this provider.

        Returns:
            List of model identifiers supported by this provider
        """
        ...

    def to_dict(self) -> dict:
        """Convert provider to dictionary representation.

        Returns:
            Dictionary with provider metadata
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

    def __str__(self) -> str:
        return self.name
