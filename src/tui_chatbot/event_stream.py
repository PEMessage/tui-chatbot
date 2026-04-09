"""Enhanced EventStream using waiters pattern (no polling timeouts).

Inspired by pi-mono's pull-based async iteration using waiters/resolvers.
Supports both async iteration (`async for`) and `await result()` promise-style.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable, Generic, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class EventStream(Generic[T, R]):
    """Dual-interface event stream: async for + await result().

    Uses waiters pattern for efficient, event-driven iteration.
    No polling timeouts - purely event-driven.

    Example:
        stream = EventStream[is_complete, extract_result]

        # Async iteration
        async for event in stream:
            print(event)

        # Promise-style result
        result = await stream.result()

    The is_complete callback detects terminal events.
    The extract_result callback computes final result from terminal event.
    """

    def __init__(
        self,
        is_complete: Callable[[T], bool],
        extract_result: Callable[[T], R],
    ):
        self._is_complete = is_complete
        self._extract_result = extract_result
        self._queue: list[T] = []
        self._waiters: list[asyncio.Future[T]] = []
        self._done = False
        self._result_future: asyncio.Future[R] = asyncio.Future()
        self._error: Optional[Exception] = None

    def push(self, event: T) -> None:
        """Push event to stream. No-op if stream is done."""
        if self._done:
            return

        # Check if this is a terminal event
        if self._is_complete(event):
            self._done = True
            if not self._result_future.done():
                self._result_future.set_result(self._extract_result(event))
            # Resolve any pending waiters with sentinel
            for waiter in self._waiters:
                if not waiter.done():
                    waiter.set_exception(StopAsyncIteration)
            self._waiters.clear()

        # Deliver to waiting consumer or queue it
        if self._waiters:
            waiter = self._waiters.pop(0)
            if not waiter.done():
                waiter.set_result(event)
        else:
            self._queue.append(event)

    def end(self, result: Optional[R] = None) -> None:
        """End stream and set result (if provided)."""
        if self._done:
            return
        self._done = True
        if result is not None and not self._result_future.done():
            self._result_future.set_result(result)
        elif not self._result_future.done():
            self._result_future.set_result(None)  # type: ignore
        # Notify all waiting consumers that we're done
        for waiter in self._waiters:
            if not waiter.done():
                waiter.set_exception(StopAsyncIteration)
        self._waiters.clear()

    def error(self, exc: Exception) -> None:
        """End stream with error."""
        if self._done:
            return
        self._done = True
        self._error = exc
        if not self._result_future.done():
            self._result_future.set_exception(exc)
        # Notify all waiting consumers
        for waiter in self._waiters:
            if not waiter.done():
                waiter.set_exception(exc)
        self._waiters.clear()

    async def result(self) -> R:
        """Get final result (promise-style)."""
        return await self._result_future

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        """Get next event (iteration-style) using waiters."""
        # Check queue first
        if self._queue:
            return self._queue.pop(0)

        # If done, stop iteration
        if self._done:
            if self._error:
                raise self._error
            raise StopAsyncIteration

        # Create a waiter and wait for an event
        waiter: asyncio.Future[T] = asyncio.Future()
        self._waiters.append(waiter)

        try:
            event = await waiter
            return event
        except StopAsyncIteration:
            raise StopAsyncIteration


class SimpleEventStream(EventStream[T, T]):
    """Simple EventStream where the terminal event is the result.

    Useful for simple cases where the last event is the result.
    """

    def __init__(self, is_complete: Callable[[T], bool]):
        super().__init__(is_complete, lambda x: x)


# Forward import for type hint without circular dependency
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .events import AssistantMessageEvent, AssistantMessage


class AssistantMessageEventStream(
    EventStream["AssistantMessageEvent", "AssistantMessage"]
):
    """EventStream for AssistantMessage events.

    Terminal events: done, error
    Result: AssistantMessage from terminal event
    """

    def __init__(self) -> None:
        from .events import is_terminal_event, extract_assistant_message

        super().__init__(is_terminal_event, extract_assistant_message)


def create_assistant_message_event_stream() -> AssistantMessageEventStream:
    """Factory function for AssistantMessageEventStream."""
    return AssistantMessageEventStream()
