"""AbortController 模式 - JavaScript 风格的取消信号."""

from __future__ import annotations

import asyncio
from typing import List, Optional, Callable, Awaitable


class AbortSignal:
    """
    取消信号 - 用于检查和等待取消状态.

    类似于 JavaScript 的 AbortSignal，可以被传递给异步操作，
    让它们能够检查是否被取消或等待取消信号。

    Attributes:
        aborted: 是否已被取消
        reason: 取消原因

    Examples:
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
        on_abort_handlers: Optional[List[Callable[[], Awaitable[None]]]] = None,
    ):
        self._event = event
        self._reason_ref = reason_ref
        self._on_abort_handlers = on_abort_handlers or []

    @property
    def aborted(self) -> bool:
        """检查是否已被取消."""
        return self._event.is_set()

    @property
    def reason(self) -> Optional[str]:
        """获取取消原因 (如果已取消)."""
        return self._reason_ref[0]

    async def wait(self) -> None:
        """等待取消信号 (如果尚未取消则阻塞)."""
        await self._event.wait()

    def throw_if_aborted(self) -> None:
        """
        如果已取消则抛出异常.

        Raises:
            asyncio.CancelledError: 如果已被取消
        """
        if self.aborted:
            raise asyncio.CancelledError(f"Aborted: {self.reason}")

    def add_event_listener(
        self, event: str, handler: Callable[[], Awaitable[None]]
    ) -> None:
        """
        添加事件监听器 (模拟 DOM 风格 API).

        Args:
            event: 事件类型 (仅支持 "abort")
            handler: 异步回调函数
        """
        if event == "abort":
            self._on_abort_handlers.append(handler)

    def remove_event_listener(
        self, event: str, handler: Callable[[], Awaitable[None]]
    ) -> None:
        """
        移除事件监听器.

        Args:
            event: 事件类型
            handler: 要移除的回调函数
        """
        if event == "abort" and handler in self._on_abort_handlers:
            self._on_abort_handlers.remove(handler)

    def __bool__(self) -> bool:
        """信号的真值 - 返回是否已取消."""
        return self.aborted

    def __repr__(self) -> str:
        """字符串表示."""
        status = f"aborted={self.aborted}"
        if self.aborted and self.reason:
            status += f", reason={self.reason!r}"
        return f"AbortSignal({status})"


class AbortController:
    """
    JavaScript 风格的取消控制器，支持超时.

    创建和管理取消信号，可以手动触发取消或设置超时自动取消。

    Attributes:
        signal: 关联的取消信号

    Examples:
        >>> # 基本用法
        >>> ctrl = AbortController()
        >>> signal = ctrl.signal
        >>> # 传递给异步操作
        >>> ctrl.abort("user cancelled")  # 触发取消

        >>> # 带超时
        >>> ctrl = AbortController(timeout=30.0)  # 30秒超时
        >>> # ... 操作完成后取消超时
        >>> ctrl.cancel_timeout()
    """

    def __init__(self, timeout: Optional[float] = None):
        """
        初始化控制器.

        Args:
            timeout: 可选的超时时间 (秒)，超时后自动触发取消
        """
        self._event = asyncio.Event()
        self._reason: List[Optional[str]] = [None]
        self._timeout_handle: Optional[asyncio.TimerHandle] = None
        self._on_abort_handlers: List[Callable[[], Awaitable[None]]] = []

        if timeout is not None and timeout > 0:
            self._set_timeout(timeout)

    def _set_timeout(self, seconds: float) -> None:
        """设置超时定时器."""
        try:
            loop = asyncio.get_running_loop()
            self._timeout_handle = loop.call_later(
                seconds, self.abort, f"timeout after {seconds}s"
            )
        except RuntimeError:
            # 没有运行中的事件循环，无法设置超时
            pass

    @property
    def signal(self) -> AbortSignal:
        """获取关联的取消信号."""
        return AbortSignal(self._event, self._reason, self._on_abort_handlers)

    def abort(self, reason: str = "aborted") -> None:
        """
        触发取消.

        Args:
            reason: 取消原因，会传递给 signal.reason

        Note:
            多次调用只有第一次有效，后续调用会被忽略。
        """
        if not self._event.is_set():
            self._reason[0] = reason
            self._event.set()

            # 取消超时定时器 (如果还在)
            if self._timeout_handle and not self._timeout_handle.cancelled():
                self._timeout_handle.cancel()
                self._timeout_handle = None

            # 调用所有 abort 监听器
            # 注意：这里不等待异步回调完成
            for handler in self._on_abort_handlers:
                try:
                    asyncio.create_task(handler())
                except RuntimeError:
                    # 没有事件循环，无法创建任务
                    pass

    def cancel_timeout(self) -> None:
        """
        取消超时定时器.

        在操作成功完成后调用，防止超时触发不必要的取消。
        """
        if self._timeout_handle:
            self._timeout_handle.cancel()
            self._timeout_handle = None

    def __repr__(self) -> str:
        """字符串表示."""
        status = "aborted" if self._event.is_set() else "active"
        if self._timeout_handle and not self._timeout_handle.cancelled():
            status += ", has_timeout"
        return f"AbortController({status})"


class AbortManager:
    """
    批量管理多个 AbortController.

    用于需要同时控制多个并发操作的场景。

    Examples:
        >>> manager = AbortManager()
        >>> ctrl1 = manager.create_controller("operation-1")
        >>> ctrl2 = manager.create_controller("operation-2", timeout=30)
        >>>
        >>> # 取消所有
        >>> manager.abort_all("shutdown")
        >>>
        >>> # 取消特定操作
        >>> manager.abort("operation-1", "timeout")
    """

    def __init__(self):
        self._controllers: Dict[str, AbortController] = {}

    def create_controller(
        self, name: str, timeout: Optional[float] = None
    ) -> AbortController:
        """
        创建并注册一个新的控制器.

        Args:
            name: 控制器名称 (用于后续引用)
            timeout: 可选的超时时间

        Returns:
            新创建的 AbortController
        """
        ctrl = AbortController(timeout)
        self._controllers[name] = ctrl
        return ctrl

    def get(self, name: str) -> Optional[AbortController]:
        """获取指定名称的控制器."""
        return self._controllers.get(name)

    def abort(self, name: str, reason: str = "aborted") -> bool:
        """
        取消特定控制器.

        Args:
            name: 控制器名称
            reason: 取消原因

        Returns:
            是否成功找到并取消
        """
        ctrl = self._controllers.get(name)
        if ctrl:
            ctrl.abort(reason)
            return True
        return False

    def abort_all(self, reason: str = "aborted") -> None:
        """取消所有控制器."""
        for ctrl in self._controllers.values():
            ctrl.abort(reason)

    def remove(self, name: str) -> bool:
        """
        移除控制器.

        Returns:
            是否成功移除
        """
        if name in self._controllers:
            del self._controllers[name]
            return True
        return False

    def clear(self) -> None:
        """清空所有控制器 (不触发取消)."""
        self._controllers.clear()

    def list_active(self) -> List[str]:
        """列出所有活跃 (未取消) 的控制器名称."""
        return [
            name for name, ctrl in self._controllers.items() if not ctrl.signal.aborted
        ]

    def __len__(self) -> int:
        """控制器数量."""
        return len(self._controllers)

    def __contains__(self, name: str) -> bool:
        """检查是否包含指定名称的控制器."""
        return name in self._controllers


# 导入 Dict 用于 AbortManager
from typing import Dict  # noqa: E402


# 导出所有公共 API
__all__ = [
    "AbortSignal",
    "AbortController",
    "AbortManager",
]
