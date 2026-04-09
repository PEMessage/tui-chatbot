"""AbortController pattern - JavaScript-style cancellation signals."""

from __future__ import annotations

import asyncio
from typing import Callable, List, Optional


class AbortSignal:
    """Cancellation signal for checking and waiting for abort state.

    Similar to JavaScript's AbortSignal, can be passed to async operations
    to allow them to check if they've been cancelled or wait for the abort signal.

    Attributes:
        aborted: Whether the operation has been cancelled
        reason: Reason for cancellation

    Example:
        >>> ctrl = AbortController()
        >>> signal = ctrl.signal
        >>>
        >>> async def long_operation(signal: AbortSignal):
        ...     while not signal.aborted:
        ...         await asyncio.sleep(0.1)
        ...     raise asyncio.CancelledError(f"Aborted: {signal.reason}")
    """

    def __init__(
        self,
        event: asyncio.Event,
        reason_ref: List[Optional[str]],
        on_abort_handlers: Optional[List[Callable[[], None]]] = None,
    ):
        self._event = event
        self._reason_ref = reason_ref
        self._on_abort_handlers = on_abort_handlers or []

    @property
    def aborted(self) -> bool:
        """Check if the operation has been cancelled."""
        return self._event.is_set()

    @property
    def reason(self) -> Optional[str]:
        """Get the reason for cancellation (if cancelled)."""
        return self._reason_ref[0]

    async def wait(self) -> None:
        """Wait for the abort signal (blocks if not yet cancelled)."""
        await self._event.wait()

    def throw_if_aborted(self) -> None:
        """Raise an exception if already cancelled.

        Raises:
            asyncio.CancelledError: If already cancelled
        """
        if self.aborted:
            raise asyncio.CancelledError(f"Aborted: {self.reason}")

    def __bool__(self) -> bool:
        """Truth value - returns whether aborted."""
        return self.aborted

    def __repr__(self) -> str:
        status = f"aborted={self.aborted}"
        if self.aborted and self.reason:
            status += f", reason={self.reason!r}"
        return f"AbortSignal({status})"


class AbortController:
    """JavaScript-style abort controller with optional timeout.

    Creates and manages an abort signal. Can be triggered manually
    or automatically after a timeout.

    Attributes:
        signal: Associated abort signal

    Example:
        >>> # Basic usage
        >>> ctrl = AbortController()
        >>> signal = ctrl.signal
        >>> # Pass to async operation
        >>> ctrl.abort("user cancelled")  # Trigger abort

        >>> # With timeout
        >>> ctrl = AbortController(timeout=30.0)  # 30 second timeout
        >>> # ... after operation completes, cancel timeout
        >>> ctrl.cancel_timeout()
    """

    def __init__(self, timeout: Optional[float] = None):
        """Initialize the controller.

        Args:
            timeout: Optional timeout in seconds, after which abort is triggered
        """
        self._event = asyncio.Event()
        self._reason: List[Optional[str]] = [None]
        self._timeout_handle: Optional[asyncio.TimerHandle] = None
        self._on_abort_handlers: List[Callable[[], None]] = []

        if timeout is not None and timeout > 0:
            self._set_timeout(timeout)

    def _set_timeout(self, seconds: float) -> None:
        """Set the timeout timer."""
        try:
            loop = asyncio.get_running_loop()
            self._timeout_handle = loop.call_later(
                seconds, self.abort, f"timeout after {seconds}s"
            )
        except RuntimeError:
            # No running event loop, can't set timeout
            pass

    @property
    def signal(self) -> AbortSignal:
        """Get the associated abort signal."""
        return AbortSignal(self._event, self._reason, self._on_abort_handlers)

    def abort(self, reason: str = "aborted") -> None:
        """Trigger abort.

        Args:
            reason: Reason for abort, passed to signal.reason

        Note:
            Multiple calls only have effect the first time, subsequent calls are ignored.
        """
        if not self._event.is_set():
            self._reason[0] = reason
            self._event.set()

            # Cancel timeout timer (if still running)
            if self._timeout_handle and not self._timeout_handle.cancelled():
                self._timeout_handle.cancel()
                self._timeout_handle = None

            # Call all abort listeners
            for handler in self._on_abort_handlers:
                try:
                    handler()
                except Exception:
                    # Ignore errors in handlers
                    pass

    def cancel_timeout(self) -> None:
        """Cancel the timeout timer.

        Call this after the operation completes successfully to prevent
        unnecessary abort from timeout.
        """
        if self._timeout_handle:
            self._timeout_handle.cancel()
            self._timeout_handle = None

    def __repr__(self) -> str:
        status = "aborted" if self._event.is_set() else "active"
        if self._timeout_handle and not self._timeout_handle.cancelled():
            status += ", has_timeout"
        return f"AbortController({status})"


__all__ = [
    "AbortSignal",
    "AbortController",
]
