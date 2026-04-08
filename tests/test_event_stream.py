"""Tests for EventStream module.

测试覆盖:
- EventStream 基本推送和迭代
- 双接口 (async for + await result)
- 错误处理
- 结束条件
- 并发访问
- 泛型类型支持
"""

import asyncio
import pytest
from typing import Optional

from tui_chatbot.core.event_stream import EventStream


class TestEventStreamBasic:
    """测试 EventStream 基本功能."""

    @pytest.mark.asyncio
    async def test_push_and_iterate(self):
        """测试推送和迭代."""
        stream = EventStream[str, str]()
        stream.push("event1")
        stream.push("event2")
        stream.push("event3")
        stream.end("final")

        events = []
        async for ev in stream:
            events.append(ev)

        assert events == ["event1", "event2", "event3"]

    @pytest.mark.asyncio
    async def test_result_after_iteration(self):
        """测试迭代后获取结果."""
        stream = EventStream[str, str]()
        stream.push("a")
        stream.push("b")
        stream.end("result")

        # 先迭代
        events = []
        async for ev in stream:
            events.append(ev)

        # 再获取结果
        result = await stream.result()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_result_without_iteration(self):
        """测试不迭代直接获取结果 (Promise 风格)."""
        stream = EventStream[str, int]()

        async def producer():
            await asyncio.sleep(0.01)
            stream.push("event")
            stream.end(42)

        asyncio.create_task(producer())

        # 直接等待结果
        result = await stream.result()
        assert result == 42

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """测试空流."""
        stream = EventStream[str, str]()
        stream.end("done")

        events = []
        async for ev in stream:
            events.append(ev)

        assert events == []
        result = await stream.result()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_result_only_no_events(self):
        """测试只有结果没有事件."""
        stream = EventStream[str, dict]()
        stream.end({"status": "ok"})

        events = []
        async for ev in stream:
            events.append(ev)

        assert events == []
        result = await stream.result()
        assert result == {"status": "ok"}


class TestEventStreamAsyncProducer:
    """测试异步生产者模式."""

    @pytest.mark.asyncio
    async def test_async_producer(self):
        """测试异步生产者."""
        stream = EventStream[int, str]()

        async def producer():
            for i in range(5):
                stream.push(i)
                await asyncio.sleep(0.01)
            stream.end("complete")

        task = asyncio.create_task(producer())

        events = []
        async for ev in stream:
            events.append(ev)

        await task
        assert events == [0, 1, 2, 3, 4]
        result = await stream.result()
        assert result == "complete"

    @pytest.mark.asyncio
    async def test_slow_producer(self):
        """测试慢速生产者."""
        stream = EventStream[str, str]()

        async def producer():
            stream.push("first")
            await asyncio.sleep(0.05)
            stream.push("second")
            await asyncio.sleep(0.05)
            stream.end("done")

        asyncio.create_task(producer())

        events = []
        async for ev in stream:
            events.append(ev)

        assert events == ["first", "second"]


class TestEventStreamErrorHandling:
    """测试错误处理."""

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """测试错误传播."""
        stream = EventStream[str, str]()
        stream.push("before")
        stream.error(ValueError("test error"))

        events = []
        async for ev in stream:
            events.append(ev)

        assert events == ["before"]

        with pytest.raises(ValueError, match="test error"):
            await stream.result()

    @pytest.mark.asyncio
    async def test_error_without_events(self):
        """测试直接报错."""
        stream = EventStream[str, str]()
        stream.error(RuntimeError("immediate error"))

        events = []
        async for ev in stream:
            events.append(ev)

        assert events == []

        with pytest.raises(RuntimeError, match="immediate error"):
            await stream.result()

    @pytest.mark.asyncio
    async def test_error_after_end_ignored(self):
        """测试结束后报错被忽略."""
        stream = EventStream[str, str]()
        stream.push("data")
        stream.end("result")
        stream.error(ValueError("late error"))  # 应该被忽略

        events = []
        async for ev in stream:
            events.append(ev)

        assert events == ["data"]
        result = await stream.result()
        assert result == "result"  # 不是错误


class TestEventStreamConcurrent:
    """测试并发访问."""

    @pytest.mark.asyncio
    async def test_concurrent_result_access(self):
        """测试并发获取结果."""
        stream = EventStream[str, int]()

        async def producer():
            for i in range(3):
                stream.push(f"msg{i}")
                await asyncio.sleep(0.01)
            stream.end(42)

        asyncio.create_task(producer())

        # 多个并发的 result() 调用
        results = await asyncio.gather(
            stream.result(),
            stream.result(),
            stream.result(),
        )

        assert all(r == 42 for r in results)

    @pytest.mark.asyncio
    async def test_multiple_consumers_iteration(self):
        """测试多个消费者迭代."""
        stream = EventStream[int, int]()
        stream.push(1)
        stream.push(2)
        stream.push(3)
        stream.end(100)

        # 第一个消费者
        events1 = []
        async for ev in stream:
            events1.append(ev)

        assert events1 == [1, 2, 3]

        # 后续迭代应该为空（队列为空）
        events2 = []
        async for ev in stream:
            events2.append(ev)

        assert events2 == []  # 队列为空


class TestEventStreamGenericTypes:
    """测试泛型类型."""

    @pytest.mark.asyncio
    async def test_string_stream(self):
        """测试字符串流."""
        stream = EventStream[str, str]()
        stream.push("hello")
        stream.push("world")
        stream.end("result")

        events = []
        async for ev in stream:
            events.append(ev)

        assert events == ["hello", "world"]

    @pytest.mark.asyncio
    async def test_int_stream(self):
        """测试整数流."""
        stream = EventStream[int, int]()
        stream.push(1)
        stream.push(2)
        stream.push(3)
        stream.end(100)

        events = []
        async for ev in stream:
            events.append(ev)

        assert events == [1, 2, 3]
        assert await stream.result() == 100

    @pytest.mark.asyncio
    async def test_dict_stream(self):
        """测试字典流."""
        stream = EventStream[dict, dict]()
        stream.push({"type": "a"})
        stream.push({"type": "b"})
        stream.end({"status": "done"})

        events = []
        async for ev in stream:
            events.append(ev)

        assert events == [{"type": "a"}, {"type": "b"}]
        assert await stream.result() == {"status": "done"}

    @pytest.mark.asyncio
    async def test_custom_class_stream(self):
        """测试自定义类流."""

        class Chunk:
            def __init__(self, text: str, index: int):
                self.text = text
                self.index = index

        stream = EventStream[Chunk, str]()
        stream.push(Chunk("hello", 0))
        stream.push(Chunk("world", 1))
        stream.end("complete")

        events = []
        async for ev in stream:
            events.append(ev)

        assert len(events) == 2
        assert events[0].text == "hello"
        assert events[1].index == 1


class TestEventStreamNoneResult:
    """测试 None 结果."""

    @pytest.mark.asyncio
    async def test_none_result(self):
        """测试 None 作为结果."""
        stream = EventStream[str, Optional[None]]()
        stream.push("a")
        stream.end(None)

        events = []
        async for ev in stream:
            events.append(ev)

        assert events == ["a"]
        result = await stream.result()
        assert result is None


class TestEventStreamEndPredicate:
    """测试结束条件回调."""

    @pytest.mark.asyncio
    async def test_end_predicate_auto_end(self):
        """测试结束条件自动结束."""

        def is_end_event(event):
            return event == "END"

        def extract_result(event):
            return "extracted"

        stream = EventStream[str, str](
            end_predicate=is_end_event,
            result_extractor=extract_result,
        )

        stream.push("event1")
        stream.push("event2")
        stream.push("END")  # 这应该触发自动结束

        events = []
        async for ev in stream:
            events.append(ev)

        # END 事件被推送了
        assert "END" in events
        # 流已经结束
        result = await stream.result()
        assert result == "extracted"

    @pytest.mark.asyncio
    async def test_push_after_done_ignored(self):
        """测试结束后推送被忽略."""
        stream = EventStream[str, str]()
        stream.push("a")
        stream.end("result")
        stream.push("b")  # 应该被忽略

        events = []
        async for ev in stream:
            events.append(ev)

        assert events == ["a"]  # b 被忽略了


class TestEventStreamEdgeCases:
    """测试边界情况."""

    @pytest.mark.asyncio
    async def test_immediate_end(self):
        """测试立即结束."""
        stream = EventStream[str, str]()
        stream.end("immediate")

        events = []
        async for ev in stream:
            events.append(ev)

        assert events == []
        assert await stream.result() == "immediate"

    @pytest.mark.asyncio
    async def test_single_event(self):
        """测试单事件."""
        stream = EventStream[str, str]()
        stream.push("only")
        stream.end("done")

        events = []
        async for ev in stream:
            events.append(ev)

        assert events == ["only"]

    @pytest.mark.asyncio
    async def test_many_events(self):
        """测试大量事件."""
        stream = EventStream[int, int]()
        count = 1000

        for i in range(count):
            stream.push(i)
        stream.end(count)

        events = []
        async for ev in stream:
            events.append(ev)

        assert len(events) == count
        assert events == list(range(count))

    @pytest.mark.asyncio
    async def test_result_type_different(self):
        """测试事件类型和结果类型不同."""
        stream = EventStream[str, int]()
        stream.push("a")
        stream.push("b")
        stream.end(42)

        events = []
        async for ev in stream:
            events.append(ev)

        assert events == ["a", "b"]
        assert await stream.result() == 42
