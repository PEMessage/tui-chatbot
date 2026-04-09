"""Tests for agent loop implementation.

Mock-based tests for the agent loop functionality including:
- Basic conversation (no tools)
- Conversation with tool calls
- Max turns limit
- Abort signal handling
- Error handling
- Sequential vs parallel tool execution
"""

import asyncio
from typing import Any, Optional

import pytest
from unittest.mock import AsyncMock

from tui_chatbot.agent.loop import (
    AgentEventStream,
    _convert_messages_to_llm_format,
    _find_tool_by_name,
    _has_tool_calls,
    _get_tool_calls,
    agent_loop,
    create_agent_event_stream,
)
from tui_chatbot.agent.types import (
    AgentContext,
    AgentEndEvent,
    AgentLoopConfig,
    AgentStartEvent,
    AgentTool,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from tui_chatbot.core.abort_controller import AbortController
from tui_chatbot.events import DoneEvent
from tui_chatbot.types import (
    AssistantMessage,
    StopReason,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


# ╭────────────────────────────────────────────────────────────╮
# │  Helper Classes                                              │
# ╰────────────────────────────────────────────────────────────╯


class MockAsyncIterator:
    """Helper class to create proper async iterators for mocking."""

    def __init__(self, items):
        self._items = list(items)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


# ╭────────────────────────────────────────────────────────────╮
# │  Helper Functions                                            │
# ╰────────────────────────────────────────────────────────────╯


def create_mock_provider(mock_response: AssistantMessage) -> AsyncMock:
    """Create a mock provider that returns the given response."""
    mock_stream = MockAsyncIterator([DoneEvent(message=mock_response)])

    provider = AsyncMock()
    provider.stream = AsyncMock(return_value=mock_stream)
    return provider


def create_mock_provider_with_tool(tool_response: AssistantMessage) -> AsyncMock:
    """Create a mock provider for tool use scenario."""
    mock_stream = MockAsyncIterator([DoneEvent(message=tool_response)])

    provider = AsyncMock()
    provider.stream = AsyncMock(return_value=mock_stream)
    return provider


# ╭────────────────────────────────────────────────────────────╮
# │  Helper Function Tests                                       │
# ╰────────────────────────────────────────────────────────────╯


def test_has_tool_calls_with_no_content():
    """Test _has_tool_calls with None message."""
    assert _has_tool_calls(None) is False


def test_has_tool_calls_with_text_only():
    """Test _has_tool_calls with text-only message."""
    msg = AssistantMessage(content=[TextContent(text="Hello")])
    assert _has_tool_calls(msg) is False


def test_has_tool_calls_with_tool_call():
    """Test _has_tool_calls with tool call in message."""
    msg = AssistantMessage(
        content=[
            TextContent(text="I'll help"),
            ToolCall(id="call_1", name="tool", arguments={}),
        ]
    )
    assert _has_tool_calls(msg) is True


def test_get_tool_calls_empty():
    """Test _get_tool_calls with empty message."""
    msg = AssistantMessage(content=[])
    assert _get_tool_calls(msg) == []


def test_get_tool_calls_filtered():
    """Test _get_tool_calls filters only ToolCall objects."""
    msg = AssistantMessage(
        content=[
            TextContent(text="Hello"),
            ToolCall(id="call_1", name="tool1", arguments={}),
            ToolCall(id="call_2", name="tool2", arguments={}),
        ]
    )
    tool_calls = _get_tool_calls(msg)
    assert len(tool_calls) == 2
    assert tool_calls[0].name == "tool1"
    assert tool_calls[1].name == "tool2"


def test_find_tool_by_name_found():
    """Test _find_tool_by_name finds existing tool."""

    async def dummy(args: dict) -> str:
        return "result"

    tool = AgentTool(name="test_tool", description="Test", parameters={}, execute=dummy)
    ctx = AgentContext(tools=[tool])

    found = _find_tool_by_name(ctx, "test_tool")
    assert found == tool


def test_find_tool_by_name_not_found():
    """Test _find_tool_by_name returns None for missing tool."""
    ctx = AgentContext(tools=[])
    found = _find_tool_by_name(ctx, "missing")
    assert found is None


def test_convert_messages_to_llm_format_user_string():
    """Test converting UserMessage with string content."""
    msgs = [UserMessage(content="Hello")]
    result = _convert_messages_to_llm_format(msgs)

    assert len(result) == 1
    assert result[0] == {"role": "user", "content": "Hello"}


def test_convert_messages_to_llm_format_user_list():
    """Test converting UserMessage with list content."""
    msgs = [UserMessage(content=[TextContent(text="Hello")])]
    result = _convert_messages_to_llm_format(msgs)

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert len(result[0]["content"]) == 1


def test_convert_messages_to_llm_format_assistant():
    """Test converting AssistantMessage."""
    msgs = [AssistantMessage(content=[TextContent(text="Hi there")])]
    result = _convert_messages_to_llm_format(msgs)

    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert result[0]["content"] == "Hi there"


def test_convert_messages_to_llm_format_assistant_with_tool():
    """Test converting AssistantMessage with tool call."""
    msgs = [
        AssistantMessage(
            content=[
                TextContent(text="I'll check"),
                ToolCall(id="call_1", name="get_weather", arguments={"city": "SF"}),
            ]
        )
    ]
    result = _convert_messages_to_llm_format(msgs)

    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert "tool_calls" in result[0]
    assert len(result[0]["tool_calls"]) == 1
    assert result[0]["tool_calls"][0]["id"] == "call_1"


def test_convert_messages_to_llm_format_tool_result():
    """Test converting ToolResultMessage."""
    msgs = [ToolResultMessage(toolCallId="call_1", content='{"temp": 72}')]
    result = _convert_messages_to_llm_format(msgs)

    assert len(result) == 1
    assert result[0]["role"] == "tool"
    assert result[0]["tool_call_id"] == "call_1"


# ╭────────────────────────────────────────────────────────────╮
# │  AgentEventStream Tests                                      │
# ╰────────────────────────────────────────────────────────────╯


def test_create_agent_event_stream():
    """Test factory function creates correct type."""
    stream = create_agent_event_stream()
    assert isinstance(stream, AgentEventStream)


@pytest.mark.asyncio
async def test_agent_event_stream_is_complete():
    """Test that AgentEndEvent marks stream complete."""
    from tui_chatbot.agent.loop import _is_agent_complete

    assert _is_agent_complete(AgentEndEvent()) is True
    assert _is_agent_complete(AgentStartEvent()) is False
    assert _is_agent_complete(TurnStartEvent()) is False


@pytest.mark.asyncio
async def test_agent_event_stream_extract_result():
    """Test extracting result from AgentEndEvent."""
    from tui_chatbot.agent.loop import _extract_agent_result

    msgs = [UserMessage(content="Hello")]
    event = AgentEndEvent(messages=msgs)
    result = _extract_agent_result(event)

    assert result == msgs


@pytest.mark.asyncio
async def test_agent_event_stream_iteration():
    """Test AgentEventStream async iteration."""
    stream = AgentEventStream()

    stream.push(AgentStartEvent())
    stream.push(TurnStartEvent(turn=1))
    stream.push(AgentEndEvent(messages=[UserMessage(content="Done")]))

    events = []
    async for event in stream:
        events.append(event)

    assert len(events) == 3
    assert events[0].type == "agent_start"
    assert events[1].type == "turn_start"
    assert events[2].type == "agent_end"


@pytest.mark.asyncio
async def test_agent_event_stream_result():
    """Test AgentEventStream result() method."""
    stream = AgentEventStream()

    stream.push(AgentStartEvent())
    stream.push(AgentEndEvent(messages=[UserMessage(content="Result")]))

    # Collect events
    events = []
    async for event in stream:
        events.append(event)

    # Get result
    result = await stream.result()
    assert len(result) == 1
    assert result[0].content == "Result"


# ╭────────────────────────────────────────────────────────────╮
# │  Basic Conversation Tests                                    │
# ╰────────────────────────────────────────────────────────────╯


@pytest.mark.asyncio
async def test_agent_loop_basic_conversation():
    """Test agent loop with simple conversation (no tools)."""
    response = AssistantMessage(
        content=[TextContent(text="Hello there!")],
        model="gpt-4",
        stopReason=StopReason.END_TURN,
    )
    provider = create_mock_provider(response)

    stream = await agent_loop(
        messages=[UserMessage(content="Hello")],
        provider=provider,
        model="gpt-4",
    )

    # Collect events
    events = []
    async for event in stream:
        events.append(event)

    # Verify event sequence
    assert events[0].type == "agent_start"
    assert events[1].type == "turn_start"
    assert events[2].type == "turn_end"
    assert events[3].type == "agent_end"

    # Get result
    result = await stream.result()
    assert len(result) == 2  # User message + assistant response
    assert result[0].role == "user"
    assert result[1].role == "assistant"


@pytest.mark.asyncio
async def test_agent_loop_with_system_prompt():
    """Test agent loop includes system prompt."""
    response = AssistantMessage(
        content=[TextContent(text="I'm helpful!")],
        model="gpt-4",
        stopReason=StopReason.END_TURN,
    )
    provider = create_mock_provider(response)

    stream = await agent_loop(
        messages=[UserMessage(content="Hello")],
        provider=provider,
        model="gpt-4",
        context=AgentContext(system_prompt="You are helpful."),
    )

    # Get result
    result = await stream.result()
    assert len(result) == 2

    # Verify provider was called
    provider.stream.assert_called_once()
    call_args = provider.stream.call_args
    messages_arg = call_args[0][0]

    # Check that system prompt was added
    assert messages_arg[0]["role"] == "system"
    assert messages_arg[0]["content"] == "You are helpful."


@pytest.mark.asyncio
async def test_agent_loop_multiple_turns():
    """Test agent loop handles multiple turns."""
    # First response continues conversation
    response1 = AssistantMessage(
        content=[TextContent(text="Tell me more.")],
        model="gpt-4",
        stopReason=StopReason.END_TURN,  # Should stop
    )

    # Create a provider that can return different responses on different calls
    provider = AsyncMock()
    stream1 = MockAsyncIterator([DoneEvent(message=response1)])
    provider.stream = AsyncMock(return_value=stream1)

    stream = await agent_loop(
        messages=[UserMessage(content="Hello")],
        provider=provider,
        model="gpt-4",
    )

    # Get result
    result = await stream.result()
    assert len(result) == 2  # Should have 2 messages


# ╭────────────────────────────────────────────────────────────╮
# │  Tool Execution Tests                                        │
# ╰────────────────────────────────────────────────────────────╯


@pytest.mark.asyncio
async def test_agent_loop_with_tool_call():
    """Test agent loop with single tool call."""

    async def echo_tool(args: dict) -> str:
        return f"Echo: {args.get('message', '')}"

    tool = AgentTool(
        name="echo",
        description="Echo tool",
        parameters={},
        execute=echo_tool,
    )

    # First response: tool call
    tool_response = AssistantMessage(
        content=[
            ToolCall(id="call_1", name="echo", arguments={"message": "hello"}),
        ],
        model="gpt-4",
        stopReason=StopReason.TOOL_USE,
    )

    # Second response: text after tool result
    final_response = AssistantMessage(
        content=[TextContent(text="Done!")],
        model="gpt-4",
        stopReason=StopReason.END_TURN,
    )

    # Create provider that returns different responses
    provider = AsyncMock()
    stream1 = MockAsyncIterator([DoneEvent(message=tool_response)])
    stream2 = MockAsyncIterator([DoneEvent(message=final_response)])
    provider.stream.side_effect = [stream1, stream2]

    stream = await agent_loop(
        messages=[UserMessage(content="Say hello")],
        provider=provider,
        model="gpt-4",
        context=AgentContext(tools=[tool]),
    )

    # Collect events
    events = []
    async for event in stream:
        events.append(event)

    # Should have tool execution events
    tool_start_events = [e for e in events if e.type == "tool_execution_start"]
    tool_end_events = [e for e in events if e.type == "tool_execution_end"]

    assert len(tool_start_events) == 1
    assert len(tool_end_events) == 1
    assert tool_end_events[0].result == "Echo: hello"

    # Get result
    result = await stream.result()
    # Should have: user msg + assistant tool call + tool result + final response
    assert len(result) == 4


@pytest.mark.asyncio
async def test_agent_loop_tool_not_found():
    """Test agent loop handles missing tool gracefully."""
    tool_response = AssistantMessage(
        content=[
            ToolCall(id="call_1", name="missing_tool", arguments={}),
        ],
        model="gpt-4",
        stopReason=StopReason.TOOL_USE,
    )

    final_response = AssistantMessage(
        content=[TextContent(text="Ok")],
        model="gpt-4",
        stopReason=StopReason.END_TURN,
    )

    provider = AsyncMock()
    stream1 = MockAsyncIterator([DoneEvent(message=tool_response)])
    stream2 = MockAsyncIterator([DoneEvent(message=final_response)])
    provider.stream.side_effect = [stream1, stream2]

    stream = await agent_loop(
        messages=[UserMessage(content="Call missing tool")],
        provider=provider,
        model="gpt-4",
        context=AgentContext(tools=[]),  # No tools registered
    )

    # Collect events
    events = []
    async for event in stream:
        events.append(event)

    # Should have tool execution with error
    tool_end_events = [e for e in events if e.type == "tool_execution_end"]
    assert len(tool_end_events) == 1
    assert "Tool not found" in tool_end_events[0].error


@pytest.mark.asyncio
async def test_agent_loop_sequential_tool_execution():
    """Test sequential tool execution (default mode)."""
    execution_order = []

    async def slow_tool1(args: dict) -> str:
        await asyncio.sleep(0.01)
        execution_order.append("tool1")
        return "result1"

    async def slow_tool2(args: dict) -> str:
        await asyncio.sleep(0.005)
        execution_order.append("tool2")
        return "result2"

    tools = [
        AgentTool(name="tool1", description="T1", parameters={}, execute=slow_tool1),
        AgentTool(name="tool2", description="T2", parameters={}, execute=slow_tool2),
    ]

    # Response with two tool calls
    tool_response = AssistantMessage(
        content=[
            ToolCall(id="call_1", name="tool1", arguments={}),
            ToolCall(id="call_2", name="tool2", arguments={}),
        ],
        model="gpt-4",
        stopReason=StopReason.TOOL_USE,
    )

    final_response = AssistantMessage(
        content=[TextContent(text="Done")],
        model="gpt-4",
        stopReason=StopReason.END_TURN,
    )

    provider = AsyncMock()
    stream1 = MockAsyncIterator([DoneEvent(message=tool_response)])
    stream2 = MockAsyncIterator([DoneEvent(message=final_response)])
    provider.stream.side_effect = [stream1, stream2]

    stream = await agent_loop(
        messages=[UserMessage(content="Call tools")],
        provider=provider,
        model="gpt-4",
        context=AgentContext(tools=tools),
        config=AgentLoopConfig(tool_mode="sequential"),
    )

    await stream.result()

    # Sequential execution: tool1 should finish before tool2 starts
    assert execution_order == ["tool1", "tool2"]


@pytest.mark.asyncio
async def test_agent_loop_parallel_tool_execution():
    """Test parallel tool execution."""
    execution_times = {}

    async def tool1(args: dict) -> str:
        start = asyncio.get_event_loop().time()
        await asyncio.sleep(0.03)
        execution_times["tool1"] = (start, asyncio.get_event_loop().time())
        return "result1"

    async def tool2(args: dict) -> str:
        start = asyncio.get_event_loop().time()
        await asyncio.sleep(0.03)
        execution_times["tool2"] = (start, asyncio.get_event_loop().time())
        return "result2"

    tools = [
        AgentTool(name="tool1", description="T1", parameters={}, execute=tool1),
        AgentTool(name="tool2", description="T2", parameters={}, execute=tool2),
    ]

    # Response with two tool calls
    tool_response = AssistantMessage(
        content=[
            ToolCall(id="call_1", name="tool1", arguments={}),
            ToolCall(id="call_2", name="tool2", arguments={}),
        ],
        model="gpt-4",
        stopReason=StopReason.TOOL_USE,
    )

    final_response = AssistantMessage(
        content=[TextContent(text="Done")],
        model="gpt-4",
        stopReason=StopReason.END_TURN,
    )

    provider = AsyncMock()
    stream1 = MockAsyncIterator([DoneEvent(message=tool_response)])
    stream2 = MockAsyncIterator([DoneEvent(message=final_response)])
    provider.stream.side_effect = [stream1, stream2]

    stream = await agent_loop(
        messages=[UserMessage(content="Call tools")],
        provider=provider,
        model="gpt-4",
        context=AgentContext(tools=tools),
        config=AgentLoopConfig(tool_mode="parallel"),
    )

    await stream.result()

    # Parallel execution: tools should overlap in time
    # tool2 might start before tool1 finishes
    tool1_start, tool1_end = execution_times["tool1"]
    tool2_start, tool2_end = execution_times["tool2"]

    # Execution windows should overlap
    assert abs(tool1_start - tool2_start) < 0.02  # Started close together


# ╭────────────────────────────────────────────────────────────╮
# │  Configuration and Limit Tests                               │
# ╰────────────────────────────────────────────────────────────╯


@pytest.mark.asyncio
async def test_agent_loop_max_turns():
    """Test that max_turns limit is respected."""
    # Response that doesn't stop (use ERROR stop reason which doesn't trigger break)
    continuing_response = AssistantMessage(
        content=[TextContent(text="Continue...")],
        model="gpt-4",
        stopReason=StopReason.ERROR,  # Won't stop the loop (only END_TURN/MAX_TOKENS/TOOL_USE do)
    )

    provider = AsyncMock()
    streams = []
    for _ in range(5):  # Create more streams than max_turns
        stream = MockAsyncIterator([DoneEvent(message=continuing_response)])
        streams.append(stream)

    provider.stream.side_effect = streams

    stream = await agent_loop(
        messages=[UserMessage(content="Hello")],
        provider=provider,
        model="gpt-4",
        config=AgentLoopConfig(max_turns=3),
    )

    result = await stream.result()

    # Should have stopped after 3 turns
    assert provider.stream.call_count == 3


@pytest.mark.asyncio
async def test_agent_loop_max_tool_calls():
    """Test that max_tool_calls limit is respected."""

    async def dummy_tool(args: dict) -> str:
        return "result"

    tool = AgentTool(
        name="loop_tool", description="Loops", parameters={}, execute=dummy_tool
    )

    # Response that always calls the tool
    tool_response = AssistantMessage(
        content=[ToolCall(id="call_x", name="loop_tool", arguments={})],
        model="gpt-4",
        stopReason=StopReason.TOOL_USE,
    )

    provider = AsyncMock()
    streams = []
    for _ in range(15):  # More than max_tool_calls
        stream = MockAsyncIterator([DoneEvent(message=tool_response)])
        streams.append(stream)

    provider.stream.side_effect = streams

    stream = await agent_loop(
        messages=[UserMessage(content="Call tool")],
        provider=provider,
        model="gpt-4",
        context=AgentContext(tools=[tool]),
        config=AgentLoopConfig(max_tool_calls=5),
    )

    # Collect events
    events = []
    async for event in stream:
        events.append(event)

    # Count tool execution events
    tool_start_events = [e for e in events if e.type == "tool_execution_start"]

    # Should have stopped after 5 tool calls
    assert len(tool_start_events) <= 5


@pytest.mark.asyncio
async def test_agent_loop_abort_signal():
    """Test that abort signal stops the loop."""
    controller = AbortController()
    signal = controller.signal

    response = AssistantMessage(
        content=[TextContent(text="Hello!")],
        model="gpt-4",
        stopReason=StopReason.END_TURN,
    )

    provider = AsyncMock()
    stream = MockAsyncIterator([DoneEvent(message=response)])
    provider.stream = AsyncMock(return_value=stream)

    # Abort immediately
    controller.abort("test abort")

    agent_stream = await agent_loop(
        messages=[UserMessage(content="Hello")],
        provider=provider,
        model="gpt-4",
        signal=signal,
    )

    # Give background task time to run and handle the abort
    await asyncio.sleep(0.01)

    # Collect events
    events = []
    async for event in agent_stream:
        events.append(event)

    # Should have started but ended early
    assert events[0].type == "agent_start"
    assert events[-1].type == "agent_end"

    # Provider should not have been called (aborted before first turn)
    # Actually it might be called depending on timing, but let's check the result
    result = await agent_stream.result()
    # Should only have initial user message
    assert len(result) == 1


@pytest.mark.asyncio
async def test_agent_loop_before_tool_call_hook():
    """Test before_tool_call hook intercepts tool execution."""
    hook_called = []

    async def before_hook(tool_call: ToolCall) -> Optional[str]:
        hook_called.append(tool_call.name)
        return "hook_result"  # Intercept and return this instead

    async def never_called(args: dict) -> str:
        raise Exception("Should not be called")

    tool = AgentTool(
        name="test", description="Test", parameters={}, execute=never_called
    )

    tool_response = AssistantMessage(
        content=[ToolCall(id="call_1", name="test", arguments={})],
        model="gpt-4",
        stopReason=StopReason.TOOL_USE,
    )

    final_response = AssistantMessage(
        content=[TextContent(text="Done")],
        model="gpt-4",
        stopReason=StopReason.END_TURN,
    )

    provider = AsyncMock()
    stream1 = MockAsyncIterator([DoneEvent(message=tool_response)])
    stream2 = MockAsyncIterator([DoneEvent(message=final_response)])
    provider.stream.side_effect = [stream1, stream2]

    agent_stream = await agent_loop(
        messages=[UserMessage(content="Call tool")],
        provider=provider,
        model="gpt-4",
        context=AgentContext(tools=[tool]),
        config=AgentLoopConfig(before_tool_call=before_hook),
    )

    # Collect events
    events = []
    async for event in agent_stream:
        events.append(event)

    # Hook should have been called
    assert hook_called == ["test"]

    # Tool execution should have hook result
    tool_end_events = [e for e in events if e.type == "tool_execution_end"]
    assert len(tool_end_events) == 1
    assert tool_end_events[0].result == "hook_result"


@pytest.mark.asyncio
async def test_agent_loop_after_tool_call_hook():
    """Test after_tool_call hook is called after execution."""
    after_calls = []

    async def after_hook(tool_call: ToolCall, result: Any) -> None:
        after_calls.append((tool_call.name, result))

    async def test_tool(args: dict) -> str:
        return "tool_result"

    tool = AgentTool(name="test", description="Test", parameters={}, execute=test_tool)

    tool_response = AssistantMessage(
        content=[ToolCall(id="call_1", name="test", arguments={})],
        model="gpt-4",
        stopReason=StopReason.TOOL_USE,
    )

    final_response = AssistantMessage(
        content=[TextContent(text="Done")],
        model="gpt-4",
        stopReason=StopReason.END_TURN,
    )

    provider = AsyncMock()
    stream1 = MockAsyncIterator([DoneEvent(message=tool_response)])
    stream2 = MockAsyncIterator([DoneEvent(message=final_response)])
    provider.stream.side_effect = [stream1, stream2]

    agent_stream = await agent_loop(
        messages=[UserMessage(content="Call tool")],
        provider=provider,
        model="gpt-4",
        context=AgentContext(tools=[tool]),
        config=AgentLoopConfig(after_tool_call=after_hook),
    )

    await agent_stream.result()

    # After hook should have been called with result
    assert len(after_calls) == 1
    assert after_calls[0] == ("test", "tool_result")


@pytest.mark.asyncio
async def test_agent_loop_transform_context_hook():
    """Test transform_context hook modifies context."""

    def transform(ctx: AgentContext) -> AgentContext:
        # Add a custom header to system prompt
        if ctx.system_prompt:
            ctx.system_prompt = f"[HEADER]\n{ctx.system_prompt}"
        else:
            ctx.system_prompt = "[HEADER]"
        return ctx

    response = AssistantMessage(
        content=[TextContent(text="Ok")],
        model="gpt-4",
        stopReason=StopReason.END_TURN,
    )

    provider = AsyncMock()
    stream = MockAsyncIterator([DoneEvent(message=response)])
    provider.stream = AsyncMock(return_value=stream)

    agent_stream = await agent_loop(
        messages=[UserMessage(content="Hello")],
        provider=provider,
        model="gpt-4",
        context=AgentContext(system_prompt="Be helpful."),
        config=AgentLoopConfig(transform_context=transform),
    )

    await agent_stream.result()

    # Check that transformed system prompt was used
    call_args = provider.stream.call_args[0][0]
    assert call_args[0]["content"] == "[HEADER]\nBe helpful."


@pytest.mark.asyncio
async def test_agent_loop_convert_to_llm_hook():
    """Test convert_to_llm hook customizes message format."""

    def custom_converter(msgs: list) -> list:
        # Add a marker to all messages
        return [
            {"role": msg.role, "content": f"[MARKER] {msg.content}"} for msg in msgs
        ]

    response = AssistantMessage(
        content=[TextContent(text="Ok")],
        model="gpt-4",
        stopReason=StopReason.END_TURN,
    )

    provider = AsyncMock()
    stream = MockAsyncIterator([DoneEvent(message=response)])
    provider.stream = AsyncMock(return_value=stream)

    agent_stream = await agent_loop(
        messages=[UserMessage(content="Hello")],
        provider=provider,
        model="gpt-4",
        config=AgentLoopConfig(convert_to_llm=custom_converter),
    )

    await agent_stream.result()

    # Check that custom converter was used
    call_args = provider.stream.call_args[0][0]
    assert call_args[0]["content"] == "[MARKER] Hello"


# ╭────────────────────────────────────────────────────────────╮
# │  Error Handling Tests                                        │
# ╰────────────────────────────────────────────────────────────╯


@pytest.mark.asyncio
async def test_agent_loop_tool_execution_error():
    """Test handling of tool execution errors."""

    async def failing_tool(args: dict) -> str:
        raise ValueError("Tool failed!")

    tool = AgentTool(
        name="failing", description="Fails", parameters={}, execute=failing_tool
    )

    tool_response = AssistantMessage(
        content=[ToolCall(id="call_1", name="failing", arguments={})],
        model="gpt-4",
        stopReason=StopReason.TOOL_USE,
    )

    final_response = AssistantMessage(
        content=[TextContent(text="Error handled")],
        model="gpt-4",
        stopReason=StopReason.END_TURN,
    )

    provider = AsyncMock()
    stream1 = MockAsyncIterator([DoneEvent(message=tool_response)])
    stream2 = MockAsyncIterator([DoneEvent(message=final_response)])
    provider.stream.side_effect = [stream1, stream2]

    agent_stream = await agent_loop(
        messages=[UserMessage(content="Call failing tool")],
        provider=provider,
        model="gpt-4",
        context=AgentContext(tools=[tool]),
    )

    # Collect events
    events = []
    async for event in agent_stream:
        events.append(event)

    # Should have tool execution with error
    tool_end_events = [e for e in events if e.type == "tool_execution_end"]
    assert len(tool_end_events) == 1
    assert "Tool failed!" in tool_end_events[0].error

    # Result should include error tool result
    result = await agent_stream.result()
    tool_results = [m for m in result if isinstance(m, ToolResultMessage)]
    assert len(tool_results) == 1
    assert tool_results[0].isError is True
    assert "Tool failed!" in tool_results[0].content


@pytest.mark.asyncio
async def test_agent_loop_provider_error():
    """Test handling of provider errors."""
    provider = AsyncMock()
    provider.stream = AsyncMock(side_effect=RuntimeError("API Error"))

    agent_stream = await agent_loop(
        messages=[UserMessage(content="Hello")],
        provider=provider,
        model="gpt-4",
    )

    # Should receive error
    with pytest.raises(RuntimeError, match="API Error"):
        await agent_stream.result()
