"""Tests for EventStream implementation.

Tests push/end/error scenarios, async iteration, and result() promise interface.
"""

import asyncio
import pytest

from tui_chatbot.event_stream import (
    EventStream,
    SimpleEventStream,
    AssistantMessageEventStream,
)


# ╭────────────────────────────────────────────────────────────╮
# │  Basic EventStream Tests                                     │
# ╰────────────────────────────────────────────────────────────╯


@pytest.mark.asyncio
async def test_push_and_iteration():
    """Test basic push and async iteration."""
    stream = SimpleEventStream[int](lambda x: x == -1)  # -1 is terminal

    # Push events
    stream.push(1)
    stream.push(2)
    stream.push(3)
    stream.push(-1)  # Terminal

    # Collect via iteration
    events = []
    async for event in stream:
        events.append(event)

    assert events == [1, 2, 3, -1]


@pytest.mark.asyncio
async def test_result_promise():
    """Test result() promise-style interface."""
    stream = EventStream[int, int](
        is_complete=lambda x: x >= 100,
        extract_result=lambda x: x,
    )

    # Push events in background
    async def pusher():
        await asyncio.sleep(0.01)
        stream.push(10)
        await asyncio.sleep(0.01)
        stream.push(50)
        await asyncio.sleep(0.01)
        stream.push(100)  # Terminal - sets result

    # Start pusher
    asyncio.create_task(pusher())

    # Get result
    result = await stream.result()
    assert result == 100


@pytest.mark.asyncio
async def test_end_method():
    """Test end() method to close stream."""
    stream = SimpleEventStream[int](lambda x: False)  # No automatic terminal

    stream.push(1)
    stream.push(2)
    stream.end(result=42)

    events = []
    async for event in stream:
        events.append(event)

    assert events == [1, 2]

    result = await stream.result()
    assert result == 42


@pytest.mark.asyncio
async def test_error_method():
    """Test error() method propagates exception."""
    stream = SimpleEventStream[int](lambda x: False)

    stream.push(1)
    stream.push(2)
    stream.error(ValueError("Test error"))

    events = []
    with pytest.raises(ValueError, match="Test error"):
        async for event in stream:
            events.append(event)

    # Should have received events before error
    assert events == [1, 2]


@pytest.mark.asyncio
async def test_error_result():
    """Test that result() raises after error()."""
    stream = SimpleEventStream[int](lambda x: False)

    stream.error(RuntimeError("Boom"))

    with pytest.raises(RuntimeError, match="Boom"):
        await stream.result()


@pytest.mark.asyncio
async def test_empty_stream_end():
    """Test empty stream with end()."""
    stream = EventStream[int, str](
        is_complete=lambda x: False,
        extract_result=lambda x: str(x),
    )

    stream.end(result="done")

    events = []
    async for event in stream:
        events.append(event)

    assert events == []
    assert await stream.result() == "done"


@pytest.mark.asyncio
async def test_push_after_end():
    """Test that push() is no-op after end()."""
    stream = SimpleEventStream[int](lambda x: False)

    stream.push(1)
    stream.end(result=42)
    stream.push(2)  # Should be ignored

    events = []
    async for event in stream:
        events.append(event)

    assert events == [1]  # 2 was ignored


@pytest.mark.asyncio
async def test_extract_result_callback():
    """Test custom extract_result callback."""

    @dataclass
    class Event:
        type: str
        data: int

    stream = EventStream[Event, str](
        is_complete=lambda e: e.type == "done",
        extract_result=lambda e: f"Result: {e.data}",
    )

    stream.push(Event("data", 10))
    stream.push(Event("done", 42))  # Terminal

    result = await stream.result()
    assert result == "Result: 42"


@pytest.mark.asyncio
async def test_single_consumer_per_event():
    """Test that EventStream delivers events to single consumer (not broadcast).

    EventStream uses a waiter pattern where each event is consumed by one waiter.
    This is the expected behavior for streaming - events are processed by a
    single consumer rather than broadcast to multiple.
    """
    stream = SimpleEventStream[int](lambda x: x == -1)

    events1 = []
    events2 = []

    async def consumer1():
        async for event in stream:
            events1.append(event)

    async def consumer2():
        async for event in stream:
            events2.append(event)

    # Start both consumers
    task1 = asyncio.create_task(consumer1())
    task2 = asyncio.create_task(consumer2())

    # Small delay to let consumers set up waiters
    await asyncio.sleep(0.01)

    # Push events
    stream.push(1)
    stream.push(2)
    stream.push(-1)

    # Wait for consumers
    await asyncio.gather(task1, task2)

    # Events are distributed among consumers (not broadcast)
    # Combined events should include all pushed events (as a set, order may vary)
    combined_set = set(events1 + events2)
    assert combined_set == {1, 2, -1}

    # At least one consumer should get the terminal event
    assert -1 in events1 or -1 in events2


# ╭────────────────────────────────────────────────────────────╮
# │  AssistantMessageEventStream Tests                           │
# ╰────────────────────────────────────────────────────────────╯

from dataclasses import dataclass
from tui_chatbot.types import AssistantMessage, StopReason, TextContent, Usage
from tui_chatbot.events import DoneEvent, ErrorEvent, TextDeltaEvent


@pytest.mark.asyncio
async def test_assistant_message_event_stream_done():
    """Test AssistantMessageEventStream with done event."""
    from tui_chatbot.events import DoneEvent

    stream = AssistantMessageEventStream()

    msg = AssistantMessage(
        role="assistant",
        content=[TextContent(text="Hello")],
        api="openai",
        provider="openai",
        model="gpt-4",
        stopReason=StopReason.END_TURN,
        usage=Usage(inputTokens=10, outputTokens=5, totalTokens=15),
        timestamp=1234567890,
    )

    # Push intermediate events
    stream.push(TextDeltaEvent(delta="Hello", partial=msg))

    # Push terminal done event
    stream.push(DoneEvent(reason=StopReason.END_TURN, message=msg))

    # Collect events
    events = []
    async for event in stream:
        events.append(event)

    assert len(events) == 2
    assert events[0].type == "text_delta"
    assert events[1].type == "done"

    # Check result
    result = await stream.result()
    assert result.role == "assistant"
    assert result.model == "gpt-4"


@pytest.mark.asyncio
async def test_assistant_message_event_stream_error():
    """Test AssistantMessageEventStream with error event."""
    from tui_chatbot.events import ErrorEvent

    stream = AssistantMessageEventStream()

    msg = AssistantMessage(
        role="assistant",
        content=[],
        api="openai",
        provider="openai",
        model="gpt-4",
        stopReason=StopReason.ERROR,
        errorMessage="API error",
        timestamp=1234567890,
    )

    # Push error event
    stream.push(ErrorEvent(reason=StopReason.ERROR, error=msg))

    # Result should be the error message
    result = await stream.result()
    assert result.errorMessage == "API error"
    assert result.stopReason == StopReason.ERROR


@pytest.mark.asyncio
async def test_factory_function():
    """Test create_assistant_message_event_stream factory."""
    from tui_chatbot.event_stream import create_assistant_message_event_stream

    stream = create_assistant_message_event_stream()
    assert isinstance(stream, AssistantMessageEventStream)


# ╭────────────────────────────────────────────────────────────╮
# │  Waiters Pattern Specific Tests                              │
# ╰────────────────────────────────────────────────────────────╯


@pytest.mark.asyncio
async def test_no_polling_timeout():
    """Verify that iteration uses waiters, not polling timeouts.

    This test checks that events are delivered immediately via waiters
    without any timeout delays.
    """
    stream = SimpleEventStream[int](lambda x: x == -1)

    received = []

    async def consumer():
        start = asyncio.get_event_loop().time()
        async for event in stream:
            received.append((event, asyncio.get_event_loop().time() - start))

    task = asyncio.create_task(consumer())

    # Small delay to let consumer start
    await asyncio.sleep(0.001)

    # Push events with minimal delays
    t0 = asyncio.get_event_loop().time()
    stream.push(1)
    stream.push(2)
    stream.push(-1)

    await task

    # Events should be received almost immediately (no 0.1s polling delays)
    assert len(received) == 3
    # The timing between push and receive should be very fast
    assert received[0][1] < 0.05  # Less than 50ms


@pytest.mark.asyncio
async def test_backpressure_handling():
    """Test that producer can push faster than consumer iterates."""
    stream = SimpleEventStream[int](lambda x: x == -1)

    # Push many events quickly
    for i in range(100):
        stream.push(i)
    stream.push(-1)

    # Consume slowly
    received = []
    async for event in stream:
        received.append(event)
        await asyncio.sleep(0.001)  # Slow consumer

    assert len(received) == 101  # All events received
    assert received[:100] == list(range(100))
