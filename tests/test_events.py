"""Tests for events module.

Tests event construction and union type checking.
"""

import pytest
from dataclasses import asdict, fields

from tui_chatbot.events import (
    StartEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThinkingStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    DoneEvent,
    ErrorEvent,
    AssistantMessageEvent,
    is_terminal_event,
    extract_assistant_message,
)
from tui_chatbot.types import (
    AssistantMessage,
    StopReason,
    TextContent,
    ThinkingContent,
    ToolCall,
    Usage,
)


# ╭────────────────────────────────────────────────────────────╮
# │  Event Construction Tests                                    │
# ╰────────────────────────────────────────────────────────────╯


def test_start_event_defaults():
    """Test StartEvent with default values."""
    event = StartEvent()
    assert event.type == "start"
    assert isinstance(event.partial, AssistantMessage)


def test_start_event_custom():
    """Test StartEvent with custom values."""
    msg = AssistantMessage(model="gpt-4")
    event = StartEvent(partial=msg)
    assert event.type == "start"
    assert event.partial.model == "gpt-4"


def test_text_start_event():
    """Test TextStartEvent construction."""
    msg = AssistantMessage(model="gpt-4")
    event = TextStartEvent(content_index=0, partial=msg)
    assert event.type == "text_start"
    assert event.content_index == 0


def test_text_delta_event():
    """Test TextDeltaEvent construction."""
    msg = AssistantMessage(model="gpt-4")
    event = TextDeltaEvent(content_index=0, delta="Hello", partial=msg)
    assert event.type == "text_delta"
    assert event.delta == "Hello"


def test_text_end_event():
    """Test TextEndEvent construction."""
    msg = AssistantMessage(model="gpt-4")
    content = TextContent(text="Hello world")
    event = TextEndEvent(content_index=0, content=content, partial=msg)
    assert event.type == "text_end"
    assert event.content.text == "Hello world"


def test_thinking_start_event():
    """Test ThinkingStartEvent construction."""
    msg = AssistantMessage(model="o1-preview")
    event = ThinkingStartEvent(content_index=0, partial=msg)
    assert event.type == "thinking_start"


def test_thinking_delta_event():
    """Test ThinkingDeltaEvent construction."""
    msg = AssistantMessage(model="o1-preview")
    event = ThinkingDeltaEvent(content_index=0, delta="Let me think...", partial=msg)
    assert event.type == "thinking_delta"
    assert event.delta == "Let me think..."


def test_thinking_end_event():
    """Test ThinkingEndEvent construction."""
    msg = AssistantMessage(model="o1-preview")
    content = ThinkingContent(thinking="Analysis complete")
    event = ThinkingEndEvent(content_index=0, content=content, partial=msg)
    assert event.type == "thinking_end"
    assert event.content.thinking == "Analysis complete"


def test_toolcall_start_event():
    """Test ToolCallStartEvent construction."""
    msg = AssistantMessage(model="gpt-4")
    event = ToolCallStartEvent(content_index=1, partial=msg)
    assert event.type == "toolcall_start"
    assert event.content_index == 1


def test_toolcall_delta_event():
    """Test ToolCallDeltaEvent construction."""
    msg = AssistantMessage(model="gpt-4")
    event = ToolCallDeltaEvent(content_index=1, delta='{"city": "SF"}', partial=msg)
    assert event.type == "toolcall_delta"


def test_toolcall_end_event():
    """Test ToolCallEndEvent construction."""
    msg = AssistantMessage(model="gpt-4")
    tool_call = ToolCall(id="call_1", name="get_weather", arguments={"city": "SF"})
    event = ToolCallEndEvent(content_index=1, tool_call=tool_call, partial=msg)
    assert event.type == "toolcall_end"
    assert event.tool_call.name == "get_weather"


def test_done_event():
    """Test DoneEvent construction."""
    msg = AssistantMessage(
        model="gpt-4",
        stopReason=StopReason.END_TURN,
        content=[TextContent(text="Result")],
    )
    event = DoneEvent(reason=StopReason.END_TURN, message=msg)
    assert event.type == "done"
    assert event.reason == StopReason.END_TURN
    assert event.message.stopReason == StopReason.END_TURN


def test_error_event():
    """Test ErrorEvent construction."""
    msg = AssistantMessage(
        model="gpt-4",
        stopReason=StopReason.ERROR,
        errorMessage="API error",
    )
    event = ErrorEvent(reason=StopReason.ERROR, error=msg)
    assert event.type == "error"
    assert event.reason == StopReason.ERROR
    assert event.error.errorMessage == "API error"


# ╭────────────────────────────────────────────────────────────╮
# │  Union Type Tests                                            │
# ╰────────────────────────────────────────────────────────────╯


def test_event_union_accepts_all_types():
    """Test that AssistantMessageEvent accepts all event variants."""
    events: list[AssistantMessageEvent] = [
        StartEvent(),
        TextStartEvent(),
        TextDeltaEvent(),
        TextEndEvent(),
        ThinkingStartEvent(),
        ThinkingDeltaEvent(),
        ThinkingEndEvent(),
        ToolCallStartEvent(),
        ToolCallDeltaEvent(),
        ToolCallEndEvent(),
        DoneEvent(),
        ErrorEvent(),
    ]

    assert len(events) == 12
    types = [e.type for e in events]
    assert "start" in types
    assert "done" in types
    assert "error" in types


def test_event_type_field_access():
    """Test accessing type field on union type."""
    event: AssistantMessageEvent = TextDeltaEvent()

    # All events have type field
    assert event.type == "text_delta"

    # Can check type
    if event.type == "text_delta":
        assert isinstance(event, TextDeltaEvent)


def test_event_isinstance_checking():
    """Test isinstance checking on union type."""
    events: list[AssistantMessageEvent] = [
        StartEvent(),
        TextDeltaEvent(delta="Hello"),
        DoneEvent(),
    ]

    delta_events = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert len(delta_events) == 1
    assert delta_events[0].delta == "Hello"


# ╭────────────────────────────────────────────────────────────╮
# │  Terminal Event Helper Tests                                 │
# ╰────────────────────────────────────────────────────────────╯


def test_is_terminal_event_done():
    """Test is_terminal_event with done event."""
    event = DoneEvent()
    assert is_terminal_event(event) is True


def test_is_terminal_event_error():
    """Test is_terminal_event with error event."""
    event = ErrorEvent()
    assert is_terminal_event(event) is True


def test_is_terminal_event_non_terminal():
    """Test is_terminal_event with non-terminal events."""
    non_terminal_events = [
        StartEvent(),
        TextStartEvent(),
        TextDeltaEvent(),
        TextEndEvent(),
        ThinkingStartEvent(),
        ThinkingDeltaEvent(),
        ThinkingEndEvent(),
        ToolCallStartEvent(),
        ToolCallDeltaEvent(),
        ToolCallEndEvent(),
    ]

    for event in non_terminal_events:
        assert is_terminal_event(event) is False, f"{event.type} should not be terminal"


def test_extract_assistant_message_from_done():
    """Test extract_assistant_message from DoneEvent."""
    msg = AssistantMessage(model="gpt-4", content=[TextContent(text="Hello")])
    event = DoneEvent(message=msg)

    result = extract_assistant_message(event)
    assert result.model == "gpt-4"
    assert result.content[0].text == "Hello"


def test_extract_assistant_message_from_error():
    """Test extract_assistant_message from ErrorEvent."""
    msg = AssistantMessage(
        model="gpt-4",
        stopReason=StopReason.ERROR,
        errorMessage="Something went wrong",
    )
    event = ErrorEvent(error=msg)

    result = extract_assistant_message(event)
    assert result.errorMessage == "Something went wrong"


def test_extract_assistant_message_raises_on_non_terminal():
    """Test extract_assistant_message raises on non-terminal event."""
    event = TextDeltaEvent()

    with pytest.raises(ValueError, match="Unexpected event type"):
        extract_assistant_message(event)


# ╭────────────────────────────────────────────────────────────╮
# │  Event Protocol Flow Tests                                     │
# ╰────────────────────────────────────────────────────────────╯


def test_text_event_flow():
    """Test complete text content event flow."""
    msg = AssistantMessage(model="gpt-4")

    flow: list[AssistantMessageEvent] = [
        StartEvent(partial=msg),
        TextStartEvent(content_index=0, partial=msg),
        TextDeltaEvent(content_index=0, delta="Hello", partial=msg),
        TextDeltaEvent(content_index=0, delta=" ", partial=msg),
        TextDeltaEvent(content_index=0, delta="world", partial=msg),
        TextEndEvent(
            content_index=0, content=TextContent(text="Hello world"), partial=msg
        ),
        DoneEvent(reason=StopReason.END_TURN, message=msg),
    ]

    # Verify flow ends with terminal event
    assert is_terminal_event(flow[-1])
    assert extract_assistant_message(flow[-1]) is msg


def test_thinking_event_flow():
    """Test complete thinking content event flow."""
    msg = AssistantMessage(model="o1-preview")

    flow: list[AssistantMessageEvent] = [
        StartEvent(partial=msg),
        ThinkingStartEvent(content_index=0, partial=msg),
        ThinkingDeltaEvent(content_index=0, delta="Analyzing...", partial=msg),
        ThinkingEndEvent(
            content_index=0,
            content=ThinkingContent(thinking="Analyzing..."),
            partial=msg,
        ),
        TextStartEvent(content_index=1, partial=msg),
        TextDeltaEvent(content_index=1, delta="Result", partial=msg),
        TextEndEvent(content_index=1, content=TextContent(text="Result"), partial=msg),
        DoneEvent(reason=StopReason.END_TURN, message=msg),
    ]

    assert len(flow) == 8
    assert flow[1].type == "thinking_start"
    assert flow[4].type == "text_start"


def test_toolcall_event_flow():
    """Test complete tool call event flow."""
    msg = AssistantMessage(model="gpt-4")

    flow: list[AssistantMessageEvent] = [
        StartEvent(partial=msg),
        TextStartEvent(content_index=0, partial=msg),
        TextDeltaEvent(content_index=0, delta="I'll check that", partial=msg),
        TextEndEvent(
            content_index=0, content=TextContent(text="I'll check that"), partial=msg
        ),
        ToolCallStartEvent(content_index=1, partial=msg),
        ToolCallDeltaEvent(content_index=1, delta='{"city": "', partial=msg),
        ToolCallDeltaEvent(content_index=1, delta='SF"}', partial=msg),
        ToolCallEndEvent(
            content_index=1,
            tool_call=ToolCall(
                id="call_1", name="get_weather", arguments={"city": "SF"}
            ),
            partial=msg,
        ),
        DoneEvent(reason=StopReason.TOOL_USE, message=msg),
    ]

    terminal = flow[-1]
    assert isinstance(terminal, DoneEvent)
    assert terminal.reason == StopReason.TOOL_USE


# ╭────────────────────────────────────────────────────────────╮
# │  Serialization Tests                                         │
# ╰────────────────────────────────────────────────────────────╯


def test_text_delta_event_asdict():
    """Test TextDeltaEvent serialization."""
    msg = AssistantMessage(model="gpt-4")
    event = TextDeltaEvent(content_index=0, delta="Hello", partial=msg)

    data = asdict(event)
    assert data["type"] == "text_delta"
    assert data["content_index"] == 0
    assert data["delta"] == "Hello"
    assert data["partial"]["model"] == "gpt-4"


def test_done_event_asdict():
    """Test DoneEvent serialization."""
    msg = AssistantMessage(
        model="gpt-4",
        stopReason=StopReason.END_TURN,
        usage=Usage(inputTokens=10, outputTokens=5),
    )
    event = DoneEvent(reason=StopReason.END_TURN, message=msg)

    data = asdict(event)
    assert data["type"] == "done"
    assert data["reason"] == "endTurn"
    assert data["message"]["model"] == "gpt-4"
    assert data["message"]["usage"]["inputTokens"] == 10


# ╭────────────────────────────────────────────────────────────╮
# │  Event Field Tests                                             │
# ╰────────────────────────────────────────────────────────────╯


def test_all_events_have_type_field():
    """Verify all event dataclasses have type field."""
    event_classes = [
        StartEvent,
        TextStartEvent,
        TextDeltaEvent,
        TextEndEvent,
        ThinkingStartEvent,
        ThinkingDeltaEvent,
        ThinkingEndEvent,
        ToolCallStartEvent,
        ToolCallDeltaEvent,
        ToolCallEndEvent,
        DoneEvent,
        ErrorEvent,
    ]

    for cls in event_classes:
        field_names = [f.name for f in fields(cls)]
        assert "type" in field_names, f"{cls.__name__} missing type field"


def test_delta_events_have_content_index():
    """Verify delta events have content_index field."""
    delta_classes = [
        TextDeltaEvent,
        ThinkingDeltaEvent,
        ToolCallDeltaEvent,
    ]

    for cls in delta_classes:
        field_names = [f.name for f in fields(cls)]
        assert "content_index" in field_names, f"{cls.__name__} missing content_index"
        assert "delta" in field_names, f"{cls.__name__} missing delta"


def test_delta_events_have_partial():
    """Verify all events have partial field except terminal events."""
    non_terminal_classes = [
        StartEvent,
        TextStartEvent,
        TextDeltaEvent,
        TextEndEvent,
        ThinkingStartEvent,
        ThinkingDeltaEvent,
        ThinkingEndEvent,
        ToolCallStartEvent,
        ToolCallDeltaEvent,
        ToolCallEndEvent,
    ]

    for cls in non_terminal_classes:
        field_names = [f.name for f in fields(cls)]
        assert "partial" in field_names, f"{cls.__name__} missing partial field"
