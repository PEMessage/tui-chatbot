"""High-level API interface using the new provider system.

The Daemon provides a simplified interface for chat interactions,
using the provider registry to get the appropriate provider for
configured model. It maintains conversation history and provides
structured event streaming.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, List, Optional

from .config import Config
from .event_stream import AssistantMessageEventStream
from .types import AssistantMessage, TextContent, UserMessage
from .agent.types import AgentMessage

if TYPE_CHECKING:
    from .core.abort_controller import AbortSignal
    from .events import AssistantMessageEvent
    from .providers import Provider


SYSTEM_MSG_CONTENT = "You are a helpful assistant."


class Daemon:
    """High-level API interface using the new provider system.

    Maintains conversation state and provides streaming chat interface.
    Uses the provider registry to automatically resolve and use the
    appropriate provider based on the configured model.

    Example:
        config = Config(api_key="sk-...", model="gpt-4")
        daemon = Daemon(config)

        # Stream chat response
        stream = daemon.chat("Hello!")
        async for event in stream:
            if event.type == "text_delta":
                print(event.delta, end="")

        result = await stream.result()
    """

    def __init__(self, config: Config):
        self.config = config
        self.provider: Optional[Provider] = None
        self.model: str = config.model
        self.messages: List[AgentMessage] = []

        # Initialize messages with system prompt
        self.clear()

        # Initialize provider based on config
        if config.api_key:
            self._init_provider()

    def _init_provider(self) -> None:
        """Initialize the provider based on configuration."""
        from .models import get_model_or_default
        from .providers import get_for_model, create_provider_from_env

        # Get model configuration to find provider name
        model_config = get_model_or_default(self.model)

        try:
            # Try to get provider from registry
            self.provider = get_for_model(self.model)
        except KeyError:
            # Try to create from environment
            self.provider = create_provider_from_env(model_config.provider)

    def _ensure_provider(self) -> Optional[Provider]:
        """Ensure provider is initialized.

        Returns:
            Provider instance or None if not available
        """
        if self.provider is None and self.config.api_key:
            self._init_provider()
        return self.provider

    def _trim_history(self) -> None:
        """Trim conversation history to keep within limits."""
        max_len = self.config.history + 1  # +1 for system message
        if len(self.messages) > max_len:
            # Keep system (first) and most recent messages
            self.messages = [self.messages[0]] + self.messages[-self.config.history :]

    def chat(
        self,
        text: str,
        signal: Optional[AbortSignal] = None,
    ) -> AssistantMessageEventStream:
        """Generate events from chat using provider.

        Adds user message, streams assistant response, and adds
        the complete response to history.

        Args:
            text: User input text
            signal: Optional abort signal for cancellation

        Returns:
            EventStream for iteration and result
        """
        stream = AssistantMessageEventStream()

        provider = self._ensure_provider()
        if provider is None:
            from .events import ErrorEvent
            from .types import StopReason

            error_msg = AssistantMessage(
                role="assistant",
                content=[TextContent(text="Error: No API key configured")],
                stopReason=StopReason.ERROR,
                errorMessage="No API key configured",
            )
            stream.push(ErrorEvent(reason=StopReason.ERROR, error=error_msg))
            return stream

        # Trim history and add user message
        self._trim_history()
        user_msg = UserMessage(content=text, timestamp=int(time.time()))
        self.messages.append(user_msg)

        async def _stream() -> None:
            try:
                # Stream from provider
                provider_stream = await provider.stream(
                    self.messages,
                    self.model,
                    signal=signal,
                )

                final_message: Optional[AssistantMessage] = None

                async for event in provider_stream:
                    # Forward event to our stream
                    stream.push(event)

                    # Track final message from terminal events
                    if event.type == "done":
                        final_message = event.message
                    elif event.type == "error":
                        final_message = event.error

                # Add final message to history if available
                if final_message:
                    self.messages.append(final_message)

            except Exception as e:
                from .events import ErrorEvent
                from .types import StopReason

                error_msg = AssistantMessage(
                    role="assistant",
                    content=[TextContent(text=f"Error: {str(e)}")],
                    stopReason=StopReason.ERROR,
                    errorMessage=str(e),
                )
                stream.push(ErrorEvent(reason=StopReason.ERROR, error=error_msg))

        # Start streaming task
        import asyncio

        asyncio.create_task(_stream())
        return stream

    def switch_model(self, name: str) -> None:
        """Switch to a different model (may change provider).

        Args:
            name: New model identifier
        """
        self.model = name

        # Re-initialize provider for new model
        self.provider = None
        if self.config.api_key:
            self._init_provider()

    def clear(self) -> None:
        """Clear conversation history (keeping system prompt)."""
        from .types import TextContent

        system_msg = AssistantMessage(
            role="assistant",
            content=[TextContent(text=SYSTEM_MSG_CONTENT)],
            timestamp=int(time.time()),
        )
        self.messages = [system_msg]

    async def list_models(self) -> List[str]:
        """List available models from current provider.

        Returns:
            List of model identifiers
        """
        provider = self._ensure_provider()
        if provider is not None:
            try:
                return await provider.list_models()
            except Exception:
                pass  # Fall through to defaults

        # Return default models from model registry
        from .models import list_models

        return list_models()
