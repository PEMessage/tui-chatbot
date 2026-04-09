"""Event protocol for streaming assistant messages.

Event flow:
    start -> text_start -> text_delta* -> text_end
          -> thinking_start -> thinking_delta* -> thinking_end
          -> toolcall_start -> toolcall_delta* -> toolcall_end
          -> done | error

Each event carries:
    - type: Event type name
    - partial: Current AssistantMessage snapshot
    - content_index: Index of content block being updated
    - delta: Incremental change (for delta events)
    - content/tool_call: Final value (for end events)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from .types import (
    AssistantMessage,
    StopReason,
    TextContent,
    ThinkingContent,
    ToolCall,
    Usage,
)


# ╭────────────────────────────────────────────────────────────╮
# │  Event Types                                                 │
# ╰────────────────────────────────────────────────────────────╯


@dataclass
class StartEvent:
    """Stream start event."""

    type: str = "start"
    partial: AssistantMessage = field(default_factory=lambda: AssistantMessage())


@dataclass
class TextStartEvent:
    """Text block start event."""

    type: str = "text_start"
    content_index: int = 0
    partial: AssistantMessage = field(default_factory=lambda: AssistantMessage())


@dataclass
class TextDeltaEvent:
    """Text incremental update event."""

    type: str = "text_delta"
    content_index: int = 0
    delta: str = ""
    partial: AssistantMessage = field(default_factory=lambda: AssistantMessage())


@dataclass
class TextEndEvent:
    """Text block completion event."""

    type: str = "text_end"
    content_index: int = 0
    content: TextContent = field(default_factory=lambda: TextContent())
    partial: AssistantMessage = field(default_factory=lambda: AssistantMessage())


@dataclass
class ThinkingStartEvent:
    """Thinking block start event."""

    type: str = "thinking_start"
    content_index: int = 0
    partial: AssistantMessage = field(default_factory=lambda: AssistantMessage())


@dataclass
class ThinkingDeltaEvent:
    """Thinking incremental update event."""

    type: str = "thinking_delta"
    content_index: int = 0
    delta: str = ""
    partial: AssistantMessage = field(default_factory=lambda: AssistantMessage())


@dataclass
class ThinkingEndEvent:
    """Thinking block completion event."""

    type: str = "thinking_end"
    content_index: int = 0
    content: ThinkingContent = field(default_factory=lambda: ThinkingContent())
    partial: AssistantMessage = field(default_factory=lambda: AssistantMessage())


@dataclass
class ToolCallStartEvent:
    """Tool call block start event."""

    type: str = "toolcall_start"
    content_index: int = 0
    partial: AssistantMessage = field(default_factory=lambda: AssistantMessage())


@dataclass
class ToolCallDeltaEvent:
    """Tool call incremental update event."""

    type: str = "toolcall_delta"
    content_index: int = 0
    delta: str = ""
    partial: AssistantMessage = field(default_factory=lambda: AssistantMessage())


@dataclass
class ToolCallEndEvent:
    """Tool call block completion event."""

    type: str = "toolcall_end"
    content_index: int = 0
    tool_call: ToolCall = field(default_factory=lambda: ToolCall())
    partial: AssistantMessage = field(default_factory=lambda: AssistantMessage())


@dataclass
class DoneEvent:
    """Stream completion event (success)."""

    type: str = "done"
    reason: StopReason = StopReason.END_TURN
    message: AssistantMessage = field(default_factory=lambda: AssistantMessage())


@dataclass
class ErrorEvent:
    """Stream completion event (error/aborted)."""

    type: str = "error"
    reason: StopReason = StopReason.ERROR
    error: AssistantMessage = field(default_factory=lambda: AssistantMessage())


# ╭────────────────────────────────────────────────────────────╮
# │  Union Type and Helpers                                      │
# ╰────────────────────────────────────────────────────────────╯

AssistantMessageEvent = Union[
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


def is_terminal_event(event: AssistantMessageEvent) -> bool:
    """Check if event is terminal (done or error)."""
    return event.type in ("done", "error")


def extract_assistant_message(event: AssistantMessageEvent) -> AssistantMessage:
    """Extract final AssistantMessage from terminal event.

    Raises:
        ValueError: If event is not a terminal event
    """
    if isinstance(event, DoneEvent):
        return event.message
    elif isinstance(event, ErrorEvent):
        return event.error
    else:
        raise ValueError(f"Unexpected event type for final result: {event.type}")
