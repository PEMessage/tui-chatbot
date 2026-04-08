"""EventStream 类 - 双接口事件流."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Generic, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class EventStream(Generic[T, R]):
    """
    双接口事件流：支持 async for 迭代和 await result() 获取结果.

    用法：
        stream = agent_loop(...)  # 返回 EventStream
        async for event in stream:
            print(event)
        result = await stream.result()
    """

    def __init__(
        self,
        end_predicate: Optional[callable] = None,
        result_extractor: Optional[callable] = None,
    ):
        self._queue: asyncio.Queue[T] = asyncio.Queue()
        self._done = asyncio.Event()
        self._result_future: asyncio.Future[R] = asyncio.Future()
        self._error: Optional[Exception] = None
        self._end_predicate = end_predicate
        self._result_extractor = result_extractor

    def push(self, event: T) -> None:
        """推送事件到流中."""
        if self._done.is_set():
            return
        self._queue.put_nowait(event)

        # 检查是否是结束事件
        if self._end_predicate and self._end_predicate(event):
            if self._result_extractor:
                result = self._result_extractor(event)
                self.end(result)

    def end(self, result: Optional[R] = None) -> None:
        """结束流并设置结果."""
        if not self._done.is_set():
            if result is not None and not self._result_future.done():
                self._result_future.set_result(result)
            elif not self._result_future.done():
                self._result_future.set_result(None)  # type: ignore
            self._done.set()

    def error(self, exc: Exception) -> None:
        """以错误结束流."""
        if not self._done.is_set():
            self._error = exc
            if not self._result_future.done():
                self._result_future.set_exception(exc)
            self._done.set()

    async def result(self) -> R:
        """获取最终结果 (Promise 风格)."""
        await self._done.wait()
        if self._error:
            raise self._error
        return await self._result_future

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        """获取下一个事件 (迭代风格)."""
        while True:
            if self._done.is_set() and self._queue.empty():
                raise StopAsyncIteration
            try:
                return self._queue.get_nowait()
            except asyncio.QueueEmpty:
                try:
                    await asyncio.wait_for(self._done.wait(), timeout=0.1)
                    if self._queue.empty():
                        raise StopAsyncIteration
                except asyncio.TimeoutError:
                    continue
