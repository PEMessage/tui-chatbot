"""Core message and content types.

Type-safe content handling with standardized metadata.
Uses dataclasses for clean, type-hinted definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union


# ╭────────────────────────────────────────────────────────────╮
# │  Content Types                                               │
# ╰────────────────────────────────────────────────────────────╯


@dataclass
class TextContent:
    """Text content block."""

    type: str = "text"
    text: str = ""


@dataclass
class ThinkingContent:
    """Thinking/reasoning content block.

    For models that expose reasoning (o1, o3, Claude, etc.)
    """

    type: str = "thinking"
    thinking: str = ""
    thinkingSignature: Optional[str] = None
    redacted: bool = False


@dataclass
class ToolCall:
    """Tool call content block (from assistant)."""

    type: str = "toolCall"
    id: str = ""
    name: str = ""
    arguments: dict = field(default_factory=dict)


@dataclass
class ImageContent:
    """Image content block.

    Source can be base64 data or URL depending on provider.
    """

    type: str = "image"
    source: dict = field(default_factory=dict)
    # Common fields for convenience
    data: Optional[str] = None  # base64 encoded
    mimeType: Optional[str] = None  # e.g., "image/jpeg"
    url: Optional[str] = None  # URL if not base64


# Content union type
Content = Union[TextContent, ThinkingContent, ToolCall, ImageContent]


# ╭────────────────────────────────────────────────────────────╮
# │  Metadata Types                                              │
# ╰────────────────────────────────────────────────────────────╯


@dataclass
class Usage:
    """Token usage statistics."""

    inputTokens: int = 0
    outputTokens: int = 0
    totalTokens: int = 0
    cost: float = 0.0

    # Extended fields for providers that support them
    cacheRead: int = 0
    cacheWrite: int = 0
    inputCost: float = 0.0
    outputCost: float = 0.0


class StopReason(str, Enum):
    """Reason why the assistant message stopped generating."""

    END_TURN = "endTurn"
    MAX_TOKENS = "maxTokens"
    TOOL_USE = "toolUse"
    ERROR = "error"
    ABORTED = "aborted"


# ╭────────────────────────────────────────────────────────────╮
# │  Message Types                                               │
# ╰────────────────────────────────────────────────────────────╯


@dataclass
class UserMessage:
    """User message in the conversation."""

    role: str = "user"
    content: Union[str, list[Content]] = ""
    timestamp: int = 0


@dataclass
class AssistantMessage:
    """Assistant message in the conversation.

    Contains full metadata about the response including
    usage, model info, and stop reason.
    """

    role: str = "assistant"
    content: list[Content] = field(default_factory=list)
    api: str = ""
    provider: str = ""
    model: str = ""
    responseId: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    stopReason: StopReason = StopReason.END_TURN
    errorMessage: Optional[str] = None
    timestamp: int = 0


@dataclass
class ToolResultMessage:
    """Tool execution result message.

    Sent back to the model after a tool call.
    """

    role: str = "tool"
    toolCallId: str = ""
    content: str = ""
    isError: bool = False
    timestamp: int = 0


# Message union type
Message = Union[UserMessage, AssistantMessage, ToolResultMessage]
