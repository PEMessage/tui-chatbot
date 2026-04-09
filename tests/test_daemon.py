"""Tests for the new Daemon using provider system."""

import asyncio
import pytest
from dataclasses import dataclass, field
from typing import List, Optional

from tui_chatbot.config import Config
from tui_chatbot.daemon import Daemon, SYSTEM_MSG_CONTENT
from tui_chatbot.types import (
    AssistantMessage,
    TextContent,
    StopReason,
    UserMessage,
)
from tui_chatbot.agent.types import AgentMessage
from tui_chatbot.events import (
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    DoneEvent,
    ErrorEvent,
)
from tui_chatbot.event_stream import AssistantMessageEventStream
from tui_chatbot.providers.base import Provider, ProviderConfig


# ╭────────────────────────────────────────────────────────────╮
# │  Mock Provider                                               │
# ╰────────────────────────────────────────────────────────────╯


class MockProvider(Provider):
    """Mock provider for testing."""

    def __init__(self, response_text: str = "Hello!"):
        self._response_text = response_text
        self._config = ProviderConfig()

    @property
    def name(self) -> str:
        return "mock"

    async def stream(self, messages, model, **kwargs):
        stream = AssistantMessageEventStream()

        async def _stream():
            # Emit simple text response
            stream.push(TextStartEvent())

            for char in self._response_text:
                stream.push(TextDeltaEvent(delta=char, partial=AssistantMessage()))

            text_content = TextContent(text=self._response_text)
            final_message = AssistantMessage(
                role="assistant",
                content=[text_content],
                model=model,
                provider=self.name,
                stopReason=StopReason.END_TURN,
            )

            stream.push(TextEndEvent(content=text_content, partial=final_message))
            stream.push(DoneEvent(reason=StopReason.END_TURN, message=final_message))

        asyncio.create_task(_stream())
        return stream

    async def list_models(self) -> List[str]:
        return ["mock-model-1", "mock-model-2"]


# ╭────────────────────────────────────────────────────────────╮
# │  Tests                                                       │
# ╰────────────────────────────────────────────────────────────╯


class TestDaemonBasic:
    """Basic Daemon tests."""

    def test_init_without_api_key(self):
        """Daemon can be initialized without API key."""
        config = Config(api_key="", model="gpt-3.5-turbo")
        daemon = Daemon(config)

        assert daemon.config == config
        assert daemon.model == "gpt-3.5-turbo"
        assert daemon.provider is None
        assert len(daemon.messages) == 1  # System message

    def test_init_with_api_key(self, monkeypatch):
        """Daemon initializes provider when API key is available."""
        from tui_chatbot.providers import register

        # Register mock provider
        register("mock", lambda: MockProvider())

        # Temporarily override model provider
        from tui_chatbot import models

        original_model = models.KNOWN_MODELS.get("gpt-3.5-turbo")
        mock_model = models.Model(
            id="gpt-3.5-turbo",
            name="GPT-3.5 Turbo",
            provider="mock",
        )
        models.KNOWN_MODELS["gpt-3.5-turbo"] = mock_model

        try:
            config = Config(api_key="test-key", model="gpt-3.5-turbo")
            daemon = Daemon(config)

            assert daemon.provider is not None
            assert daemon.provider.name == "mock"
        finally:
            # Restore original model
            if original_model:
                models.KNOWN_MODELS["gpt-3.5-turbo"] = original_model
            else:
                del models.KNOWN_MODELS["gpt-3.5-turbo"]

    def test_clear(self):
        """Clear resets messages to system message only."""
        config = Config(api_key="", model="gpt-3.5-turbo")
        daemon = Daemon(config)

        # Add some messages
        daemon.messages.append(UserMessage(content="Hello"))
        daemon.messages.append(AssistantMessage(content=[TextContent(text="Hi")]))

        assert len(daemon.messages) == 3  # system + user + assistant

        daemon.clear()

        assert len(daemon.messages) == 1  # Only system
        assert daemon.messages[0].role == "assistant"

    def test_switch_model(self):
        """Switch model updates the model and resets provider."""
        from tui_chatbot.providers import register

        register("mock", lambda: MockProvider())

        from tui_chatbot import models

        original_model = models.KNOWN_MODELS.get("gpt-3.5-turbo")
        mock_model = models.Model(
            id="gpt-3.5-turbo",
            name="GPT-3.5 Turbo",
            provider="mock",
        )
        models.KNOWN_MODELS["gpt-3.5-turbo"] = mock_model

        try:
            config = Config(api_key="test-key", model="gpt-3.5-turbo")
            daemon = Daemon(config)

            assert daemon.provider is not None

            daemon.switch_model("gpt-4")

            assert daemon.model == "gpt-4"
            assert daemon.provider is None  # Reset after switch
        finally:
            if original_model:
                models.KNOWN_MODELS["gpt-3.5-turbo"] = original_model
            else:
                del models.KNOWN_MODELS["gpt-3.5-turbo"]


class TestDaemonChat:
    """Daemon chat streaming tests."""

    @pytest.mark.asyncio
    async def test_chat_without_provider(self):
        """Chat without provider returns error event."""
        config = Config(api_key="", model="gpt-3.5-turbo")
        daemon = Daemon(config)

        stream = daemon.chat("Hello")

        events = []
        async for event in stream:
            events.append(event)

        # Should get error event
        assert len(events) == 1
        assert events[0].type == "error"

    @pytest.mark.asyncio
    async def test_chat_with_mock_provider(self, monkeypatch):
        """Chat with mock provider streams events."""
        from tui_chatbot.providers import register, clear

        clear()
        register("mock", lambda: MockProvider("Hello World"))

        from tui_chatbot import models

        original_model = models.KNOWN_MODELS.get("gpt-3.5-turbo")
        mock_model = models.Model(
            id="gpt-3.5-turbo",
            name="GPT-3.5 Turbo",
            provider="mock",
        )
        models.KNOWN_MODELS["gpt-3.5-turbo"] = mock_model

        try:
            config = Config(api_key="test-key", model="gpt-3.5-turbo")
            daemon = Daemon(config)

            stream = daemon.chat("Hello")

            events = []
            async for event in stream:
                events.append(event)

            # Should get text events and done
            assert any(e.type == "text_start" for e in events)
            assert any(e.type == "text_delta" for e in events)
            assert any(e.type == "text_end" for e in events)
            assert any(e.type == "done" for e in events)

            # Verify message added to history
            assert len(daemon.messages) == 3  # system + user + assistant

            # Verify final result
            result = await stream.result()
            assert result.role == "assistant"
            assert len(result.content) == 1
            assert result.content[0].text == "Hello World"
        finally:
            if original_model:
                models.KNOWN_MODELS["gpt-3.5-turbo"] = original_model
            else:
                del models.KNOWN_MODELS["gpt-3.5-turbo"]

    @pytest.mark.asyncio
    async def test_chat_adds_user_message(self, monkeypatch):
        """Chat adds user message to history before streaming."""
        from tui_chatbot.providers import register, clear

        clear()
        register("mock", lambda: MockProvider())

        from tui_chatbot import models

        original_model = models.KNOWN_MODELS.get("gpt-3.5-turbo")
        mock_model = models.Model(
            id="gpt-3.5-turbo",
            name="GPT-3.5 Turbo",
            provider="mock",
        )
        models.KNOWN_MODELS["gpt-3.5-turbo"] = mock_model

        try:
            config = Config(api_key="test-key", model="gpt-3.5-turbo")
            daemon = Daemon(config)

            initial_count = len(daemon.messages)

            stream = daemon.chat("Test message")
            async for _ in stream:
                pass

            # Messages: system + user + assistant response
            assert len(daemon.messages) == initial_count + 2
            # User message is at -2 position (before assistant response)
            assert daemon.messages[-2].role == "user"
            assert daemon.messages[-1].role == "assistant"
        finally:
            if original_model:
                models.KNOWN_MODELS["gpt-3.5-turbo"] = original_model
            else:
                del models.KNOWN_MODELS["gpt-3.5-turbo"]

    @pytest.mark.asyncio
    async def test_chat_history_trimming(self, monkeypatch):
        """Chat trims history when exceeding limit."""
        from tui_chatbot.providers import register, clear

        clear()
        register("mock", lambda: MockProvider())

        from tui_chatbot import models

        original_model = models.KNOWN_MODELS.get("gpt-3.5-turbo")
        mock_model = models.Model(
            id="gpt-3.5-turbo",
            name="GPT-3.5 Turbo",
            provider="mock",
        )
        models.KNOWN_MODELS["gpt-3.5-turbo"] = mock_model

        try:
            config = Config(api_key="test-key", model="gpt-3.5-turbo", history=3)
            daemon = Daemon(config)

            # Fill up history
            for i in range(10):
                daemon.messages.append(UserMessage(content=f"Message {i}"))
                daemon.messages.append(
                    AssistantMessage(content=[TextContent(text=f"Response {i}")])
                )

            initial_count = len(daemon.messages)
            assert initial_count > 4  # system + more than 3 history pairs

            # Trigger chat to cause trimming
            stream = daemon.chat("Test")
            async for _ in stream:
                pass

            # Should be trimmed to system + 3 pairs + new user + new assistant
            assert len(daemon.messages) <= 9  # system + history + new pair
        finally:
            if original_model:
                models.KNOWN_MODELS["gpt-3.5-turbo"] = original_model
            else:
                del models.KNOWN_MODELS["gpt-3.5-turbo"]


class TestDaemonListModels:
    """Daemon list_models tests."""

    @pytest.mark.asyncio
    async def test_list_models_without_provider(self):
        """List models returns defaults when no provider."""
        config = Config(api_key="", model="gpt-3.5-turbo")
        daemon = Daemon(config)

        models = await daemon.list_models()
        assert isinstance(models, list)
        # Should return defaults from models module
        assert len(models) > 0

    @pytest.mark.asyncio
    async def test_list_models_with_provider(self, monkeypatch):
        """List models delegates to provider."""
        from tui_chatbot.providers import register, clear

        clear()
        register("mock", lambda: MockProvider())

        from tui_chatbot import models as models_module

        original_model = models_module.KNOWN_MODELS.get("gpt-3.5-turbo")
        mock_model = models_module.Model(
            id="gpt-3.5-turbo",
            name="GPT-3.5 Turbo",
            provider="mock",
        )
        models_module.KNOWN_MODELS["gpt-3.5-turbo"] = mock_model

        try:
            config = Config(api_key="test-key", model="gpt-3.5-turbo")
            daemon = Daemon(config)

            models = await daemon.list_models()

            # Should delegate to mock provider
            assert models == ["mock-model-1", "mock-model-2"]
        finally:
            if original_model:
                models_module.KNOWN_MODELS["gpt-3.5-turbo"] = original_model
            else:
                del models_module.KNOWN_MODELS["gpt-3.5-turbo"]
