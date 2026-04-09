"""Tests for types module.

Tests dataclass construction and type validation.
"""

import pytest
from dataclasses import asdict

from tui_chatbot.types import (
    TextContent,
    ThinkingContent,
    ToolCall,
    ImageContent,
    Usage,
    StopReason,
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
    Content,
    Message,
)


# ╭────────────────────────────────────────────────────────────╮
# │  Content Type Tests                                          │
# ╰────────────────────────────────────────────────────────────╯


def test_text_content_defaults():
    """Test TextContent with default values."""
    content = TextContent()
    assert content.type == "text"
    assert content.text == ""


def test_text_content_custom():
    """Test TextContent with custom values."""
    content = TextContent(text="Hello world")
    assert content.type == "text"
    assert content.text == "Hello world"


def test_thinking_content_defaults():
    """Test ThinkingContent with default values."""
    content = ThinkingContent()
    assert content.type == "thinking"
    assert content.thinking == ""
    assert content.thinkingSignature is None
    assert content.redacted is False


def test_thinking_content_custom():
    """Test ThinkingContent with custom values."""
    content = ThinkingContent(
        thinking="Let me analyze...",
        thinkingSignature="sig123",
        redacted=True,
    )
    assert content.thinking == "Let me analyze..."
    assert content.thinkingSignature == "sig123"
    assert content.redacted is True


def test_tool_call_defaults():
    """Test ToolCall with default values."""
    content = ToolCall()
    assert content.type == "toolCall"
    assert content.id == ""
    assert content.name == ""
    assert content.arguments == {}


def test_tool_call_custom():
    """Test ToolCall with custom values."""
    content = ToolCall(
        id="call_123",
        name="get_weather",
        arguments={"city": "San Francisco"},
    )
    assert content.id == "call_123"
    assert content.name == "get_weather"
    assert content.arguments == {"city": "San Francisco"}


def test_image_content_defaults():
    """Test ImageContent with default values."""
    content = ImageContent()
    assert content.type == "image"
    assert content.source == {}
    assert content.data is None
    assert content.mimeType is None
    assert content.url is None


def test_image_content_base64():
    """Test ImageContent with base64 data."""
    content = ImageContent(
        source={"type": "base64", "media_type": "image/jpeg"},
        data="base64encoded...",
        mimeType="image/jpeg",
    )
    assert content.data == "base64encoded..."
    assert content.mimeType == "image/jpeg"


def test_image_content_url():
    """Test ImageContent with URL source."""
    content = ImageContent(
        source={"type": "url"},
        url="https://example.com/image.png",
    )
    assert content.url == "https://example.com/image.png"


def test_content_union():
    """Test that Content type accepts all content variants."""
    text: Content = TextContent(text="Hello")
    thinking: Content = ThinkingContent(thinking="Analysis")
    tool: Content = ToolCall(id="1", name="tool", arguments={})
    image: Content = ImageContent(data="base64...")

    assert text.type == "text"
    assert thinking.type == "thinking"
    assert tool.type == "toolCall"
    assert image.type == "image"


# ╭────────────────────────────────────────────────────────────╮
# │  Metadata Type Tests                                         │
# ╰────────────────────────────────────────────────────────────╯


def test_usage_defaults():
    """Test Usage with default values."""
    usage = Usage()
    assert usage.inputTokens == 0
    assert usage.outputTokens == 0
    assert usage.totalTokens == 0
    assert usage.cost == 0.0
    assert usage.cacheRead == 0
    assert usage.cacheWrite == 0


def test_usage_custom():
    """Test Usage with custom values."""
    usage = Usage(
        inputTokens=100,
        outputTokens=50,
        totalTokens=150,
        cost=0.002,
        cacheRead=10,
        cacheWrite=5,
    )
    assert usage.inputTokens == 100
    assert usage.outputTokens == 50
    assert usage.totalTokens == 150
    assert usage.cost == 0.002
    assert usage.cacheRead == 10
    assert usage.cacheWrite == 5


def test_stop_reason_enum():
    """Test StopReason enum values."""
    assert StopReason.END_TURN == "endTurn"
    assert StopReason.MAX_TOKENS == "maxTokens"
    assert StopReason.TOOL_USE == "toolUse"
    assert StopReason.ERROR == "error"
    assert StopReason.ABORTED == "aborted"


def test_stop_reason_comparison():
    """Test StopReason comparison."""
    reason = StopReason.TOOL_USE
    assert reason == "toolUse"
    assert reason == StopReason.TOOL_USE


# ╭────────────────────────────────────────────────────────────╮
# │  Message Type Tests                                          │
# ╰────────────────────────────────────────────────────────────╯


def test_user_message_defaults():
    """Test UserMessage with default values."""
    msg = UserMessage()
    assert msg.role == "user"
    assert msg.content == ""
    assert msg.timestamp == 0


def test_user_message_string_content():
    """Test UserMessage with string content."""
    msg = UserMessage(
        content="Hello assistant",
        timestamp=1234567890,
    )
    assert msg.content == "Hello assistant"
    assert msg.timestamp == 1234567890


def test_user_message_list_content():
    """Test UserMessage with list of content blocks."""
    msg = UserMessage(
        content=[
            TextContent(text="Look at this"),
            ImageContent(data="base64..."),
        ],
        timestamp=1234567890,
    )
    assert len(msg.content) == 2  # type: ignore
    assert msg.content[0].type == "text"  # type: ignore
    assert msg.content[1].type == "image"  # type: ignore


def test_assistant_message_defaults():
    """Test AssistantMessage with default values."""
    msg = AssistantMessage()
    assert msg.role == "assistant"
    assert msg.content == []
    assert msg.api == ""
    assert msg.provider == ""
    assert msg.model == ""
    assert msg.responseId is None
    assert isinstance(msg.usage, Usage)
    assert msg.stopReason == StopReason.END_TURN
    assert msg.errorMessage is None
    assert msg.timestamp == 0


def test_assistant_message_custom():
    """Test AssistantMessage with custom values."""
    msg = AssistantMessage(
        content=[
            ThinkingContent(thinking="Analysis..."),
            TextContent(text="Result"),
        ],
        api="openai",
        provider="openai",
        model="gpt-4",
        responseId="resp_123",
        usage=Usage(inputTokens=100, outputTokens=50),
        stopReason=StopReason.END_TURN,
        errorMessage=None,
        timestamp=1234567890,
    )
    assert len(msg.content) == 2
    assert msg.content[0].type == "thinking"
    assert msg.content[1].type == "text"
    assert msg.api == "openai"
    assert msg.model == "gpt-4"
    assert msg.responseId == "resp_123"
    assert msg.usage.inputTokens == 100


def test_assistant_message_with_tool_call():
    """Test AssistantMessage containing a tool call."""
    msg = AssistantMessage(
        content=[
            TextContent(text="I'll check the weather"),
            ToolCall(id="call_1", name="get_weather", arguments={"city": "SF"}),
        ],
        stopReason=StopReason.TOOL_USE,
    )
    assert len(msg.content) == 2
    assert msg.content[1].type == "toolCall"
    assert msg.content[1].name == "get_weather"
    assert msg.stopReason == StopReason.TOOL_USE


def test_assistant_message_error():
    """Test AssistantMessage with error state."""
    msg = AssistantMessage(
        content=[],
        stopReason=StopReason.ERROR,
        errorMessage="API rate limit exceeded",
    )
    assert msg.stopReason == StopReason.ERROR
    assert msg.errorMessage == "API rate limit exceeded"


def test_tool_result_message_defaults():
    """Test ToolResultMessage with default values."""
    msg = ToolResultMessage()
    assert msg.role == "tool"
    assert msg.toolCallId == ""
    assert msg.content == ""
    assert msg.isError is False
    assert msg.timestamp == 0


def test_tool_result_message_success():
    """Test successful ToolResultMessage."""
    msg = ToolResultMessage(
        toolCallId="call_123",
        content='{"temperature": 72}',
        isError=False,
        timestamp=1234567890,
    )
    assert msg.toolCallId == "call_123"
    assert msg.content == '{"temperature": 72}'
    assert msg.isError is False


def test_tool_result_message_error():
    """Test error ToolResultMessage."""
    msg = ToolResultMessage(
        toolCallId="call_456",
        content="Failed to fetch weather",
        isError=True,
        timestamp=1234567890,
    )
    assert msg.isError is True


def test_message_union():
    """Test that Message type accepts all message variants."""
    user: Message = UserMessage(content="Hello")
    assistant: Message = AssistantMessage(model="gpt-4")
    tool: Message = ToolResultMessage(toolCallId="1", content="result")

    assert user.role == "user"
    assert assistant.role == "assistant"
    assert tool.role == "tool"


# ╭────────────────────────────────────────────────────────────╮
# │  Serialization Tests                                           │
# ╰────────────────────────────────────────────────────────────╯


def test_text_content_asdict():
    """Test TextContent serialization."""
    content = TextContent(text="Hello")
    data = asdict(content)
    assert data == {"type": "text", "text": "Hello"}


def test_assistant_message_asdict():
    """Test AssistantMessage serialization."""
    msg = AssistantMessage(
        content=[TextContent(text="Hello")],
        model="gpt-4",
        stopReason=StopReason.END_TURN,
    )
    data = asdict(msg)
    assert data["role"] == "assistant"
    assert data["model"] == "gpt-4"
    assert data["stopReason"] == "endTurn"


def test_usage_asdict():
    """Test Usage serialization."""
    usage = Usage(inputTokens=100, outputTokens=50)
    data = asdict(usage)
    assert data == {
        "inputTokens": 100,
        "outputTokens": 50,
        "totalTokens": 0,
        "cost": 0.0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "inputCost": 0.0,
        "outputCost": 0.0,
    }


# ╭────────────────────────────────────────────────────────────╮
# │  Type Validation Tests                                       │
# ╰────────────────────────────────────────────────────────────╯


def test_content_type_field_required():
    """Test that type field is properly set."""
    text = TextContent()
    assert text.type == "text"

    # Changing type should work (though unusual)
    text.type = "custom"
    assert text.type == "custom"


def test_stop_reason_is_string_enum():
    """Test that StopReason behaves as a string enum."""
    # Can compare with string
    assert StopReason.END_TURN == "endTurn"
    # Can use in dict
    reasons = {StopReason.END_TURN: "normal", StopReason.ERROR: "failed"}
    assert reasons["endTurn"] == "normal"
