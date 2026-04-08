"""
Integration test for EventStream and AbortController with Daemon.

Tests actual usage patterns:
1. Daemon.chat() returns EventStream
2. Can iterate events and get result
3. AbortController cancels in-flight requests
4. Frontend works with new stream
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tui_chatbot.main import (
    Daemon,
    Config,
    EventStream,
    AbortController,
    Event,
    EventType,
)


async def test_daemon_chat_returns_eventstream():
    """Test: Daemon.chat() returns EventStream type."""
    print("Test 1: Daemon.chat() returns EventStream...")

    cfg = Config(api_key="fake-key")
    daemon = Daemon(cfg)

    # Without signal
    stream = daemon.chat("test")
    assert isinstance(stream, EventStream), f"Expected EventStream, got {type(stream)}"

    # With signal
    ctrl = AbortController()
    stream2 = daemon.chat("test", signal=ctrl.signal)
    assert isinstance(stream2, EventStream), (
        f"Expected EventStream, got {type(stream2)}"
    )

    print("  ✓ PASSED: Daemon.chat() returns EventStream")


async def test_daemon_chat_early_abort():
    """Test: Abort before stream starts."""
    print("Test 2: Early abort before stream...")

    cfg = Config(api_key="fake-key")
    daemon = Daemon(cfg)

    ctrl = AbortController()
    ctrl.abort("pre-flight")

    stream = daemon.chat("test", signal=ctrl.signal)

    # Collect events
    events = []
    async for ev in stream:
        events.append(ev)

    # Should get error and empty result
    assert len(events) == 1, f"Expected 1 event, got {len(events)}: {events}"
    assert events[0].type == EventType.ERROR, f"Expected ERROR, got {events[0].type}"
    assert "Aborted" in events[0].data, (
        f"Expected 'Aborted' in message, got {events[0].data}"
    )

    result = await stream.result()
    assert result == "", f"Expected empty result, got {result}"

    print("  ✓ PASSED: Early abort works")


async def test_eventstream_concurrent_usage():
    """Test: Multiple consumers can use result()."""
    print("Test 3: Concurrent result() access...")

    stream = EventStream[str, int]()

    async def producer():
        for i in range(5):
            stream.push(f"msg{i}")
            await asyncio.sleep(0.01)
        stream.end(42)

    asyncio.create_task(producer())

    # Multiple await result()
    results = await asyncio.gather(
        stream.result(),
        stream.result(),
        stream.result(),
    )

    assert all(r == 42 for r in results), f"All results should be 42, got {results}"

    print("  ✓ PASSED: Concurrent result() access works")


async def test_eventstream_error_handling():
    """Test: Error propagation through stream."""
    print("Test 4: Error propagation...")

    stream = EventStream[str, str]()
    stream.push("before")
    stream.error(ValueError("test error"))

    # Can still iterate
    events = []
    async for ev in stream:
        events.append(ev)

    assert events == ["before"], f"Expected ['before'], got {events}"

    # Error in result()
    try:
        await stream.result()
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "test error"

    print("  ✓ PASSED: Error propagation works")


async def test_stream_with_no_result():
    """Test: Stream ends without result."""
    print("Test 5: Stream without result...")

    stream = EventStream[str, None]()
    stream.push("a")
    stream.push("b")
    stream.end()  # No result

    events = []
    async for ev in stream:
        events.append(ev)

    assert events == ["a", "b"]

    result = await stream.result()
    assert result is None, f"Expected None, got {result}"

    print("  ✓ PASSED: Stream without result works")


async def test_abort_during_iteration():
    """Test: Abort signal checked during iteration."""
    print("Test 6: Abort during iteration pattern...")

    stream = EventStream[int, str]()
    ctrl = AbortController()

    async def slow_producer():
        for i in range(100):
            # Check abort signal
            if ctrl.signal.aborted:
                stream.push(-1)  # Marker for aborted
                stream.end("aborted")
                return
            stream.push(i)
            await asyncio.sleep(0.01)
        stream.end("complete")

    async def delayed_abort():
        await asyncio.sleep(0.05)
        ctrl.abort("user interrupt")

    producer_task = asyncio.create_task(slow_producer())
    abort_task = asyncio.create_task(delayed_abort())

    # Iterate
    values = []
    async for v in stream:
        values.append(v)

    await producer_task
    await abort_task

    # Should have aborted
    assert -1 in values, f"Expected abort marker -1 in {values}"
    assert len(values) < 100, f"Should have stopped early, got {len(values)} values"

    result = await stream.result()
    assert result == "aborted", f"Expected 'aborted', got {result}"

    print("  ✓ PASSED: Abort during iteration works")


async def run_all_tests():
    """Run integration tests."""
    print("=" * 60)
    print("Integration Test Suite")
    print("=" * 60)
    print()

    tests = [
        test_daemon_chat_returns_eventstream,
        test_daemon_chat_early_abort,
        test_eventstream_concurrent_usage,
        test_eventstream_error_handling,
        test_stream_with_no_result,
        test_abort_during_iteration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
