"""Tests for agent types module.

Tests dataclass construction, type validation, and event handling.
"""

import pytest
from dataclasses import asdict, is_dataclass
from typing import get_type_hints

from tui_chatbot.agent.types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentStartEvent,
    AgentTool,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from tui_chatbot.types import (
    AssistantMessage,
    StopReason,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


# ╭────────────────────────────────────────────────────────────╮
# │  AgentTool Tests                                             │
# ╰────────────────────────────────────────────────────────────╯


def test_agent_tool_is_dataclass():
    """Test that AgentTool is a dataclass."""
    assert is_dataclass(AgentTool)


def test_agent_tool_defaults():
    """Test AgentTool with default values."""

    async def dummy_execute(args: dict) -> str:
        return "result"

    tool = AgentTool(
        name="test_tool",
        description="A test tool",
        parameters={"type": "object", "properties": {}},
        execute=dummy_execute,
    )

    assert tool.name == "test_tool"
    assert tool.description == "A test tool"
    assert tool.parameters == {"type": "object", "properties": {}}
    assert tool.execute == dummy_execute


def test_agent_tool_custom_parameters():
    """Test AgentTool with custom parameters schema."""

    async def search_execute(args: dict) -> str:
        return f"Searching for {args.get('query', '')}"

    tool = AgentTool(
        name="search",
        description="Search for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                }
            },
            "required": ["query"],
        },
        execute=search_execute,
    )

    assert tool.parameters["properties"]["query"]["type"] == "string"
    assert "query" in tool.parameters["required"]


@pytest.mark.asyncio
async def test_agent_tool_execute():
    """Test that AgentTool execute can be called."""

    async def echo_execute(args: dict) -> str:
        return args.get("message", "")

    tool = AgentTool(
        name="echo",
        description="Echo tool",
        parameters={},
        execute=echo_execute,
    )

    result = await tool.execute({"message": "hello"})
    assert result == "hello"


# ╭────────────────────────────────────────────────────────────╮
# │  AgentContext Tests                                          │
# ╰────────────────────────────────────────────────────────────╯


def test_agent_context_defaults():
    """Test AgentContext with default values."""
    ctx = AgentContext()

    assert ctx.system_prompt is None
    assert ctx.messages == []
    assert ctx.tools == []


def test_agent_context_custom():
    """Test AgentContext with custom values."""
    user_msg = UserMessage(content="Hello")

    async def dummy_tool(args: dict) -> str:
        return "result"

    tool = AgentTool(
        name="test",
        description="Test tool",
        parameters={},
        execute=dummy_tool,
    )

    ctx = AgentContext(
        system_prompt="You are a helpful assistant.",
        messages=[user_msg],
        tools=[tool],
    )

    assert ctx.system_prompt == "You are a helpful assistant."
    assert len(ctx.messages) == 1
    assert ctx.messages[0].role == "user"
    assert len(ctx.tools) == 1
    assert ctx.tools[0].name == "test"


def test_agent_context_mutable_messages():
    """Test that AgentContext messages list is mutable."""
    ctx = AgentContext()

    # Should be able to append messages
    ctx.messages.append(UserMessage(content="Hello"))
    assert len(ctx.messages) == 1


def test_agent_context_mutable_tools():
    """Test that AgentContext tools list is mutable."""
    ctx = AgentContext()

    async def dummy_tool(args: dict) -> str:
        return "result"

    ctx.tools.append(
        AgentTool(
            name="test",
            description="Test",
            parameters={},
            execute=dummy_tool,
        )
    )
    assert len(ctx.tools) == 1


# ╭────────────────────────────────────────────────────────────╮
# │  AgentLoopConfig Tests                                       │
# ╰────────────────────────────────────────────────────────────╯


def test_agent_loop_config_defaults():
    """Test AgentLoopConfig with default values."""
    cfg = AgentLoopConfig()

    assert cfg.max_turns == 10
    assert cfg.max_tool_calls == 10
    assert cfg.tool_mode == "sequential"
    assert cfg.convert_to_llm is None
    assert cfg.transform_context is None
    assert cfg.before_tool_call is None
    assert cfg.after_tool_call is None


def test_agent_loop_config_custom():
    """Test AgentLoopConfig with custom values."""

    def custom_converter(msgs: list) -> list:
        return msgs

    cfg = AgentLoopConfig(
        max_turns=5,
        max_tool_calls=3,
        tool_mode="parallel",
        convert_to_llm=custom_converter,
    )

    assert cfg.max_turns == 5
    assert cfg.max_tool_calls == 3
    assert cfg.tool_mode == "parallel"
    assert cfg.convert_to_llm == custom_converter


def test_agent_loop_config_tool_modes():
    """Test that tool_mode accepts valid values."""
    sequential_cfg = AgentLoopConfig(tool_mode="sequential")
    assert sequential_cfg.tool_mode == "sequential"

    parallel_cfg = AgentLoopConfig(tool_mode="parallel")
    assert parallel_cfg.tool_mode == "parallel"


# ╭────────────────────────────────────────────────────────────╮
# │  AgentMessage Type Tests                                     │
# ╰────────────────────────────────────────────────────────────╯


def test_agent_message_union_accepts_user():
    """Test AgentMessage accepts UserMessage."""
    msg: AgentMessage = UserMessage(content="Hello")
    assert msg.role == "user"


def test_agent_message_union_accepts_assistant():
    """Test AgentMessage accepts AssistantMessage."""
    msg: AgentMessage = AssistantMessage(model="gpt-4")
    assert msg.role == "assistant"


def test_agent_message_union_accepts_tool_result():
    """Test AgentMessage accepts ToolResultMessage."""
    msg: AgentMessage = ToolResultMessage(toolCallId="123", content="result")
    assert msg.role == "tool"


# ╭────────────────────────────────────────────────────────────╮
# │  Agent Event Tests                                           │
# ╰────────────────────────────────────────────────────────────╯


def test_agent_start_event_defaults():
    """Test AgentStartEvent with default values."""
    event = AgentStartEvent()

    assert event.type == "agent_start"
    assert isinstance(event.context, AgentContext)


def test_agent_start_event_custom():
    """Test AgentStartEvent with custom context."""
    ctx = AgentContext(system_prompt="Be helpful")
    event = AgentStartEvent(context=ctx)

    assert event.type == "agent_start"
    assert event.context.system_prompt == "Be helpful"


def test_agent_end_event_defaults():
    """Test AgentEndEvent with default values."""
    event = AgentEndEvent()

    assert event.type == "agent_end"
    assert event.messages == []


def test_agent_end_event_custom():
    """Test AgentEndEvent with custom messages."""
    msgs = [
        UserMessage(content="Hello"),
        AssistantMessage(content=[TextContent(text="Hi")]),
    ]
    event = AgentEndEvent(messages=msgs)

    assert event.type == "agent_end"
    assert len(event.messages) == 2
    assert event.messages[0].role == "user"
    assert event.messages[1].role == "assistant"


def test_turn_start_event_defaults():
    """Test TurnStartEvent with default values."""
    event = TurnStartEvent()

    assert event.type == "turn_start"
    assert event.turn == 0


def test_turn_start_event_custom():
    """Test TurnStartEvent with custom turn number."""
    event = TurnStartEvent(turn=5)

    assert event.type == "turn_start"
    assert event.turn == 5


def test_turn_end_event_defaults():
    """Test TurnEndEvent with default values."""
    event = TurnEndEvent()

    assert event.type == "turn_end"
    assert event.turn == 0
    assert event.message is None


def test_turn_end_event_custom():
    """Test TurnEndEvent with custom values."""
    msg = AssistantMessage(model="gpt-4", content=[TextContent(text="Hello")])
    event = TurnEndEvent(turn=3, message=msg)

    assert event.type == "turn_end"
    assert event.turn == 3
    assert event.message is not None
    assert event.message.model == "gpt-4"


def test_tool_execution_start_event_defaults():
    """Test ToolExecutionStartEvent with default values."""
    event = ToolExecutionStartEvent()

    assert event.type == "tool_execution_start"
    assert isinstance(event.tool_call, ToolCall)


def test_tool_execution_start_event_custom():
    """Test ToolExecutionStartEvent with custom tool call."""
    tc = ToolCall(id="call_123", name="get_weather", arguments={"city": "SF"})
    event = ToolExecutionStartEvent(tool_call=tc)

    assert event.type == "tool_execution_start"
    assert event.tool_call.id == "call_123"
    assert event.tool_call.name == "get_weather"


def test_tool_execution_end_event_defaults():
    """Test ToolExecutionEndEvent with default values."""
    event = ToolExecutionEndEvent()

    assert event.type == "tool_execution_end"
    assert isinstance(event.tool_call, ToolCall)
    assert event.result is None
    assert event.error is None


def test_tool_execution_end_event_success():
    """Test ToolExecutionEndEvent with successful result."""
    tc = ToolCall(id="call_456", name="calculator")
    event = ToolExecutionEndEvent(tool_call=tc, result="42")

    assert event.type == "tool_execution_end"
    assert event.result == "42"
    assert event.error is None


def test_tool_execution_end_event_error():
    """Test ToolExecutionEndEvent with error."""
    tc = ToolCall(id="call_789", name="search")
    event = ToolExecutionEndEvent(tool_call=tc, error="Tool not found")

    assert event.type == "tool_execution_end"
    assert event.result is None
    assert event.error == "Tool not found"


# ╭────────────────────────────────────────────────────────────╮
# │  AgentEvent Union Type Tests                                 │
# ╰────────────────────────────────────────────────────────────╯


def test_agent_event_union_accepts_all_events():
    """Test AgentEvent union accepts all event types."""

    def check_event(e: AgentEvent) -> str:
        return e.type

    assert check_event(AgentStartEvent()) == "agent_start"
    assert check_event(AgentEndEvent()) == "agent_end"
    assert check_event(TurnStartEvent()) == "turn_start"
    assert check_event(TurnEndEvent()) == "turn_end"
    assert check_event(ToolExecutionStartEvent()) == "tool_execution_start"
    assert check_event(ToolExecutionEndEvent()) == "tool_execution_end"


# ╭────────────────────────────────────────────────────────────╮
# │  Serialization Tests                                         │
# ╰────────────────────────────────────────────────────────────╯


def test_agent_start_event_asdict():
    """Test AgentStartEvent serialization."""
    ctx = AgentContext(system_prompt="Test")
    event = AgentStartEvent(context=ctx)

    data = asdict(event)
    assert data["type"] == "agent_start"
    assert data["context"]["system_prompt"] == "Test"


def test_agent_end_event_asdict():
    """Test AgentEndEvent serialization."""
    msgs = [UserMessage(content="Hello")]
    event = AgentEndEvent(messages=msgs)

    data = asdict(event)
    assert data["type"] == "agent_end"
    assert len(data["messages"]) == 1


def test_tool_execution_event_asdict():
    """Test ToolExecutionEndEvent serialization."""
    tc = ToolCall(id="123", name="test", arguments={"key": "value"})
    event = ToolExecutionEndEvent(tool_call=tc, result="success")

    data = asdict(event)
    assert data["type"] == "tool_execution_end"
    assert data["tool_call"]["id"] == "123"
    assert data["tool_call"]["arguments"] == {"key": "value"}
    assert data["result"] == "success"
    assert data["error"] is None


def test_agent_loop_config_asdict():
    """Test AgentLoopConfig serialization.

    Note: Callables cannot be serialized, so they are excluded.
    """
    cfg = AgentLoopConfig(max_turns=5, tool_mode="parallel")
    data = asdict(cfg)

    assert data["max_turns"] == 5
    assert data["tool_mode"] == "parallel"
    # Callable fields are None when serialized
    assert data["convert_to_llm"] is None
    assert data["transform_context"] is None


# ╭────────────────────────────────────────────────────────────╮
# │  Type Hint Tests                                             │
# ╰────────────────────────────────────────────────────────────╯


def test_agent_tool_type_hints():
    """Test that AgentTool has correct type hints."""
    hints = get_type_hints(AgentTool)

    assert "name" in hints
    assert "description" in hints
    assert "parameters" in hints
    assert "execute" in hints


def test_agent_context_type_hints():
    """Test that AgentContext has correct type hints."""
    hints = get_type_hints(AgentContext)

    assert "system_prompt" in hints
    assert "messages" in hints
    assert "tools" in hints
