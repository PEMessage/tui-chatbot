"""Tests for Frontend event rendering."""

import asyncio
from io import StringIO
from unittest.mock import MagicMock, AsyncMock
from typing import List

import pytest

from tui_chatbot.frontend import Frontend, Colors
from tui_chatbot.daemon import Daemon
from tui_chatbot.config import Config
from tui_chatbot.types import (
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    StopReason,
    Usage,
)
from tui_chatbot.events import (
    StartEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThinkingStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    DoneEvent,
    ErrorEvent,
)


# ╭────────────────────────────────────────────────────────────╮
# │  Fixtures                                                    │
# ╰────────────────────────────────────────────────────────────╯


@pytest.fixture
def mock_daemon():
    """Create a mock daemon."""
    daemon = MagicMock(spec=Daemon)
    return daemon


@pytest.fixture
def mock_stream():
    """Create a mock event stream."""
    stream = MagicMock()
    return stream


# ╭────────────────────────────────────────────────────────────╮
# │  Event Creation Helpers                                      │
# ╰────────────────────────────────────────────────────────────╯


def create_text_events(text: str) -> List:
    """Create a sequence of text events for a response."""
    events = []
    events.append(StartEvent())
    events.append(TextStartEvent())

    for char in text:
        events.append(TextDeltaEvent(delta=char))

    text_content = TextContent(text=text)
    events.append(TextEndEvent(content=text_content))

    final_message = AssistantMessage(
        role="assistant",
        content=[text_content],
        stopReason=StopReason.END_TURN,
    )
    events.append(DoneEvent(message=final_message))

    return events


def create_thinking_events(thinking: str, text: str) -> List:
    """Create events with reasoning content."""
    events = []
    events.append(StartEvent())

    # Thinking block
    events.append(ThinkingStartEvent())
    for char in thinking:
        events.append(ThinkingDeltaEvent(delta=char))
    thinking_content = ThinkingContent(thinking=thinking)
    events.append(ThinkingEndEvent(content=thinking_content))

    # Text block
    events.append(TextStartEvent())
    for char in text:
        events.append(TextDeltaEvent(delta=char))
    text_content = TextContent(text=text)
    events.append(TextEndEvent(content=text_content))

    final_message = AssistantMessage(
        role="assistant",
        content=[thinking_content, text_content],
        stopReason=StopReason.END_TURN,
    )
    events.append(DoneEvent(message=final_message))

    return events


def create_toolcall_events(tool_name: str) -> List:
    """Create events with a tool call."""
    events = []
    events.append(StartEvent())

    # Text before tool
    events.append(TextStartEvent())
    events.append(TextDeltaEvent(delta="I'll help you with that."))
    events.append(TextEndEvent(content=TextContent(text="I'll help you with that.")))

    # Tool call
    events.append(ToolCallStartEvent())
    tool_call = ToolCall(id="123", name=tool_name, arguments={"query": "test"})
    events.append(ToolCallEndEvent(tool_call=tool_call))

    final_message = AssistantMessage(
        role="assistant",
        content=[
            TextContent(text="I'll help you with that."),
            tool_call,
        ],
        stopReason=StopReason.TOOL_USE,
    )
    events.append(DoneEvent(message=final_message))

    return events


def create_error_events(error_msg: str) -> List:
    """Create error events."""
    events = []
    events.append(StartEvent())

    error = AssistantMessage(
        role="assistant",
        content=[TextContent(text=f"Error: {error_msg}")],
        stopReason=StopReason.ERROR,
        errorMessage=error_msg,
    )
    events.append(ErrorEvent(error=error))

    return events


# ╭────────────────────────────────────────────────────────────╮
# │  Tests                                                       │
# ╰────────────────────────────────────────────────────────────╯


class TestFrontendInit:
    """Frontend initialization tests."""

    def test_init(self, mock_daemon):
        """Frontend initializes with daemon."""
        frontend = Frontend(mock_daemon)
        assert frontend.daemon == mock_daemon


class TestFrontendRenderEvents:
    """Frontend event rendering tests."""

    @pytest.mark.asyncio
    async def test_render_text_events(self, mock_daemon, capsys):
        """Text events render correctly."""
        events = create_text_events("Hello")
        mock_daemon.chat = MagicMock(return_value=create_mock_stream(events))

        frontend = Frontend(mock_daemon)
        await frontend.run(["Hi"])

        captured = capsys.readouterr()
        assert "[Assistant]:" in captured.out
        assert "Hello" in captured.out

    @pytest.mark.asyncio
    async def test_render_thinking_events(self, mock_daemon, capsys):
        """Thinking events render with gray color."""
        events = create_thinking_events("thinking", "Done!")
        mock_daemon.chat = MagicMock(return_value=create_mock_stream(events))

        frontend = Frontend(mock_daemon)
        await frontend.run(["Think"])

        captured = capsys.readouterr()
        assert "[Reasoning]:" in captured.out
        assert "Done!" in captured.out
        # Should contain gray color codes
        assert Colors.GRAY in captured.out

    @pytest.mark.asyncio
    async def test_render_toolcall_events(self, mock_daemon, capsys):
        """Tool call events render correctly."""
        events = create_toolcall_events("search")
        mock_daemon.chat = MagicMock(return_value=create_mock_stream(events))

        frontend = Frontend(mock_daemon)
        await frontend.run(["Search"])

        captured = capsys.readouterr()
        assert "[Tool:" in captured.out

    @pytest.mark.asyncio
    async def test_render_error_events(self, mock_daemon, capsys):
        """Error events render with red color."""
        events = create_error_events("API error")
        mock_daemon.chat = MagicMock(return_value=create_mock_stream(events))

        frontend = Frontend(mock_daemon)
        await frontend.run(["Error"])

        captured = capsys.readouterr()
        assert "[Error:" in captured.out
        assert "API error" in captured.out
        assert Colors.RED in captured.out

    @pytest.mark.asyncio
    async def test_render_done_with_stats(self, mock_daemon, capsys):
        """Done event with usage shows stats."""
        events = create_text_events("Hello")
        # Add usage to final message
        final_msg = events[-1].message
        final_msg.usage = Usage(
            inputTokens=10,
            outputTokens=5,
            totalTokens=15,
            cost=0.001,
        )
        mock_daemon.chat = MagicMock(return_value=create_mock_stream(events))

        frontend = Frontend(mock_daemon)
        await frontend.run(["Hi"])

        captured = capsys.readouterr()
        assert "15 tokens" in captured.out
        assert "10 in" in captured.out
        assert "5 out" in captured.out


class TestFrontendOldEvents:
    """Frontend backward compatibility tests."""

    def test_render_old_reasoning_token(self, capsys):
        """Old REASONING_TOKEN events render correctly."""
        from tui_chatbot.main import Event, EventType, C as OldColors

        frontend = Frontend(MagicMock())

        # Create old-style event
        event = Event(EventType.REASONING_TOKEN, "thinking...")
        frontend._render_event(event)

        captured = capsys.readouterr()
        assert "thinking..." in captured.out

    def test_render_old_content_token(self, capsys):
        """Old CONTENT_TOKEN events render correctly."""
        from tui_chatbot.main import Event, EventType

        frontend = Frontend(MagicMock())

        event = Event(EventType.CONTENT_TOKEN, "content")
        frontend._render_event(event)

        captured = capsys.readouterr()
        assert "content" in captured.out

    def test_render_old_error(self, capsys):
        """Old ERROR events render correctly."""
        from tui_chatbot.main import Event, EventType

        frontend = Frontend(MagicMock())

        event = Event(EventType.ERROR, "error message")
        frontend._render_event(event)

        captured = capsys.readouterr()
        assert "error message" in captured.out
        assert "Error" in captured.out


class TestFrontendPrivateMethods:
    """Frontend private method tests."""

    def test_is_old_event(self):
        """_is_old_event detects old-style events."""
        from tui_chatbot.main import Event, EventType

        frontend = Frontend(MagicMock())

        old_event = Event(EventType.CONTENT_TOKEN, "test")
        assert frontend._is_old_event(old_event) is True

        new_event = TextStartEvent()
        assert frontend._is_old_event(new_event) is False

    def test_extract_tool_name(self):
        """_extract_tool_name extracts tool name from partial message."""
        frontend = Frontend(MagicMock())

        tool_call = ToolCall(id="1", name="search", arguments={})
        partial = AssistantMessage(content=[tool_call])

        name = frontend._extract_tool_name(partial, 0)
        assert name == "search"

    def test_extract_tool_name_unknown(self):
        """_extract_tool_name returns unknown when not found."""
        frontend = Frontend(MagicMock())

        name = frontend._extract_tool_name(None, 0)
        assert name == "unknown"


# ╭────────────────────────────────────────────────────────────╮
# │  Helpers                                                     │
# ╰────────────────────────────────────────────────────────────╯


class AsyncEventIterator:
    """Helper class to mock async event iteration."""

    def __init__(self, events):
        self.events = events
        self.index = 0
        self._result_set = False
        self._result_value = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.events):
            raise StopAsyncIteration
        event = self.events[self.index]
        self.index += 1
        return event

    async def result(self):
        """Mock result method."""
        # Return the last event's message if available
        for event in reversed(self.events):
            if hasattr(event, "message"):
                return event.message
            if hasattr(event, "error"):
                return event.error
        return None


def create_mock_stream(events):
    """Create a proper mock stream that works with async for."""
    iterator = AsyncEventIterator(events)
    # Create a mock that has __aiter__ and result methods
    mock_stream = MagicMock()
    mock_stream.__aiter__ = MagicMock(return_value=iterator)
    mock_stream.result = AsyncMock(return_value=iterator.result())
    return mock_stream
