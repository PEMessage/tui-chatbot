"""
Test script for EventStream and AbortController improvements.

Tests:
1. EventStream async for iteration
2. EventStream await result() Promise-style
3. EventStream dual usage (iterate then get result)
4. AbortController abort signal
5. AbortController timeout
6. Daemon.chat() with EventStream
7. Backward compatibility (existing code still works)
"""

import asyncio
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tui_chatbot.main import (
    EventStream,
    AbortController,
    AbortSignal,
    Event,
    EventType,
)


@pytest.mark.asyncio
async def test_event_stream_basic():
    """Test: Basic EventStream push and iteration."""
    print("Test 1: EventStream basic push/iteration...")

    stream = EventStream[str, str]()
    received = []

    # Simulate producer
    def producer():
        stream.push("event1")
        stream.push("event2")
        stream.push("event3")
        stream.end("final_result")

    producer()

    # Consumer via async for
    async for event in stream:
        received.append(event)

    assert received == ["event1", "event2", "event3"], (
        f"Expected 3 events, got {received}"
    )
    result = await stream.result()
    assert result == "final_result", f"Expected 'final_result', got {result}"

    print("  ✓ PASSED: Basic iteration and result work")


@pytest.mark.asyncio
async def test_event_stream_result_first():
    """Test: Get result without iterating (promise-style)."""
    print("Test 2: EventStream promise-style result()...")

    stream = EventStream[int, int]()

    async def producer():
        await asyncio.sleep(0.01)
        for i in range(5):
            stream.push(i)
            await asyncio.sleep(0.01)
        stream.end(42)

    asyncio.create_task(producer())

    # Just await result without iterating
    result = await stream.result()

    assert result == 42, f"Expected 42, got {result}"

    print("  ✓ PASSED: Promise-style result() works")


@pytest.mark.asyncio
async def test_event_stream_dual_usage():
    """Test: Iterate then get result (dual interface)."""
    print("Test 3: EventStream dual usage (iterate + result)...")

    stream = EventStream[str, dict]()

    async def producer():
        await asyncio.sleep(0.01)
        stream.push("chunk1")
        await asyncio.sleep(0.01)
        stream.push("chunk2")
        await asyncio.sleep(0.01)
        stream.push("chunk3")
        stream.end({"status": "ok", "count": 3})

    asyncio.create_task(producer())

    # First: iterate
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert chunks == ["chunk1", "chunk2", "chunk3"], f"Expected 3 chunks, got {chunks}"

    # Then: get result
    result = await stream.result()

    assert result == {"status": "ok", "count": 3}, f"Expected dict, got {result}"

    print("  ✓ PASSED: Dual usage (iterate then result) works")


@pytest.mark.asyncio
async def test_event_stream_empty():
    """Test: Empty stream with result."""
    print("Test 4: Empty EventStream...")

    stream = EventStream[str, str]()
    stream.end("done")

    # Should not iterate any events
    events = []
    async for ev in stream:
        events.append(ev)

    assert events == [], f"Expected no events, got {events}"

    result = await stream.result()
    assert result == "done", f"Expected 'done', got {result}"

    print("  ✓ PASSED: Empty stream works")


@pytest.mark.asyncio
async def test_abort_controller_basic():
    """Test: AbortController basic abort."""
    print("Test 5: AbortController basic abort...")

    ctrl = AbortController()
    signal = ctrl.signal

    assert not signal.aborted, "Signal should not be aborted initially"

    ctrl.abort("user cancelled")

    assert signal.aborted, "Signal should be aborted"
    assert signal.reason == "user cancelled", (
        f"Expected 'user cancelled', got {signal.reason}"
    )

    print("  ✓ PASSED: Basic abort works")


@pytest.mark.asyncio
async def test_abort_controller_wait():
    """Test: AbortSignal wait() works."""
    print("Test 6: AbortSignal wait()...")

    ctrl = AbortController()
    signal = ctrl.signal

    async def aborter():
        await asyncio.sleep(0.05)
        ctrl.abort("timeout")

    asyncio.create_task(aborter())

    # Should wait until aborted
    start = asyncio.get_event_loop().time()
    await signal.wait()
    elapsed = asyncio.get_event_loop().time() - start

    assert signal.aborted, "Signal should be aborted after wait"
    assert elapsed >= 0.04, f"Expected ~0.05s delay, got {elapsed:.3f}s"

    print("  ✓ PASSED: AbortSignal wait() works")


@pytest.mark.asyncio
async def test_abort_controller_timeout():
    """Test: AbortController with timeout."""
    print("Test 7: AbortController auto timeout...")

    ctrl = AbortController(timeout=0.05)
    signal = ctrl.signal

    # Should not be aborted immediately
    assert not signal.aborted, "Should not be aborted immediately"

    # Wait for timeout
    await asyncio.sleep(0.08)

    assert signal.aborted, "Should be aborted after timeout"
    assert "timeout" in signal.reason, f"Expected timeout reason, got {signal.reason}"

    print("  ✓ PASSED: Auto timeout works")


@pytest.mark.asyncio
async def test_abort_controller_cancel_timeout():
    """Test: Cancel timeout before it fires."""
    print("Test 8: AbortController cancel timeout...")

    ctrl = AbortController(timeout=0.1)
    signal = ctrl.signal

    ctrl.cancel_timeout()
    await asyncio.sleep(0.15)

    assert not signal.aborted, "Should not be aborted after cancel"

    # Manual abort should still work
    ctrl.abort("manual")
    assert signal.aborted, "Should be aborted after manual abort"

    print("  ✓ PASSED: Cancel timeout works")


@pytest.mark.asyncio
async def test_event_stream_with_abort():
    """Test: EventStream with abort signal."""
    print("Test 9: EventStream with abort signal...")

    stream = EventStream[str, str]()
    ctrl = AbortController()

    async def producer():
        for i in range(10):
            if ctrl.signal.aborted:
                stream.push("aborted!")
                stream.end("incomplete")
                return
            stream.push(f"chunk{i}")
            await asyncio.sleep(0.01)
        stream.end("complete")

    async def aborter():
        await asyncio.sleep(0.03)  # Abort after a few chunks
        ctrl.abort("early stop")

    producer_task = asyncio.create_task(producer())
    asyncio.create_task(aborter())

    received = []
    async for chunk in stream:
        received.append(chunk)

    await producer_task

    # Should have received some chunks + aborted message
    assert "aborted!" in received, f"Expected 'aborted!' in {received}"
    assert len(received) < 10, f"Expected early termination, got {len(received)} chunks"

    result = await stream.result()
    assert result == "incomplete", f"Expected 'incomplete', got {result}"

    print("  ✓ PASSED: EventStream with abort works")


@pytest.mark.asyncio
async def test_backward_compatibility():
    """Test: Existing async for pattern still works."""
    print("Test 10: Backward compatibility (async for)...")

    stream = EventStream[Event, str]()

    # Simulate old Daemon.chat() pattern
    def old_style_producer():
        stream.push(Event(EventType.CONTENT_TOKEN, "Hello"))
        stream.push(Event(EventType.CONTENT_TOKEN, " World"))
        stream.push(Event(EventType.DONE, None))
        stream.end("Hello World")

    old_style_producer()

    # Old-style consumption via async for
    events = []
    async for ev in stream:
        events.append(ev)

    assert len(events) == 3, f"Expected 3 events, got {len(events)}"
    assert events[0].type == EventType.CONTENT_TOKEN
    assert events[0].data == "Hello"

    print("  ✓ PASSED: Backward compatibility maintained")


@pytest.mark.asyncio
async def test_event_stream_generic_types():
    """Test: EventStream with complex generic types."""
    print("Test 11: EventStream generic type hints...")

    from dataclasses import dataclass

    @dataclass
    class Chunk:
        text: str
        index: int

    @dataclass
    class Result:
        text: str
        count: int

    stream = EventStream[Chunk, Result]()

    stream.push(Chunk("hello", 0))
    stream.push(Chunk("world", 1))
    stream.end(Result("hello world", 2))

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].text == "hello"
    assert chunks[1].index == 1

    result = await stream.result()
    assert result.count == 2

    print("  ✓ PASSED: Generic types work correctly")


async def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("EventStream & AbortController Test Suite")
    print("=" * 60)
    print()

    tests = [
        test_event_stream_basic,
        test_event_stream_result_first,
        test_event_stream_dual_usage,
        test_event_stream_empty,
        test_abort_controller_basic,
        test_abort_controller_wait,
        test_abort_controller_timeout,
        test_abort_controller_cancel_timeout,
        test_event_stream_with_abort,
        test_backward_compatibility,
        test_event_stream_generic_types,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
