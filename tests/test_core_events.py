"""Tests for core events module.

测试覆盖:
- AgentEventType 枚举定义
- AgentEvent Pydantic 模型
- ChatResult 模型
- TokenStats 统计类
- 向后兼容 EventType
"""

import pytest
from datetime import datetime

from tui_chatbot.core.events import (
    AgentEventType,
    AgentEvent,
    ChatResult,
    TokenStats,
    EventType,  # 向后兼容
)


class TestAgentEventType:
    """测试 AgentEventType 枚举."""

    def test_all_event_types_exist(self):
        """测试所有预期的事件类型都存在."""
        expected_types = [
            # Agent 生命周期
            "AGENT_START",
            "AGENT_END",
            # Turn 生命周期
            "TURN_START",
            "TURN_END",
            # 消息生命周期
            "MESSAGE_START",
            "MESSAGE_UPDATE",
            "MESSAGE_END",
            # 工具生命周期
            "TOOL_EXECUTION_START",
            "TOOL_EXECUTION_UPDATE",
            "TOOL_EXECUTION_END",
            # 其他
            "ERROR",
            "STATS",
            "REASONING_TOKEN",
            "CONTENT_TOKEN",
        ]

        for type_name in expected_types:
            assert hasattr(AgentEventType, type_name), (
                f"Missing event type: {type_name}"
            )

    def test_event_type_values_are_unique(self):
        """测试所有事件类型的值都是唯一的."""
        values = [e.value for e in AgentEventType]
        assert len(values) == len(set(values)), "Event type values should be unique"

    def test_event_type_enum_iteration(self):
        """测试枚举可以正确遍历."""
        all_types = list(AgentEventType)
        assert len(all_types) >= 14  # 至少14个事件类型


class TestAgentEvent:
    """测试 AgentEvent Pydantic 模型."""

    def test_basic_event_creation(self):
        """测试创建基本事件."""
        event = AgentEvent(type=AgentEventType.AGENT_START)
        assert event.type == AgentEventType.AGENT_START
        assert event.message is None
        assert event.error is None

    def test_event_with_message(self):
        """测试带消息的事件."""
        event = AgentEvent(
            type=AgentEventType.MESSAGE_START,
            message="Hello World",
        )
        assert event.type == AgentEventType.MESSAGE_START
        assert event.message == "Hello World"

    def test_event_with_tool_info(self):
        """测试带工具信息的事件."""
        event = AgentEvent(
            type=AgentEventType.TOOL_EXECUTION_START,
            tool_call_id="call_123",
            tool_name="get_current_time",
            args={"timezone": "UTC"},
        )
        assert event.tool_call_id == "call_123"
        assert event.tool_name == "get_current_time"
        assert event.args == {"timezone": "UTC"}

    def test_event_with_error(self):
        """测试错误事件."""
        event = AgentEvent(
            type=AgentEventType.ERROR,
            error="Something went wrong",
            is_error=True,
        )
        assert event.error == "Something went wrong"
        assert event.is_error is True

    def test_event_with_progress(self):
        """测试带进度的事件."""
        event = AgentEvent(
            type=AgentEventType.TOOL_EXECUTION_UPDATE,
            progress=50,
        )
        assert event.progress == 50

    def test_progress_validation_range(self):
        """测试进度值范围验证."""
        # 有效范围 0-100
        event = AgentEvent(type=AgentEventType.TOOL_EXECUTION_UPDATE, progress=0)
        assert event.progress == 0

        event = AgentEvent(type=AgentEventType.TOOL_EXECUTION_UPDATE, progress=100)
        assert event.progress == 100

        # 超出范围应该报错
        with pytest.raises(ValueError):
            AgentEvent(type=AgentEventType.TOOL_EXECUTION_UPDATE, progress=101)

        with pytest.raises(ValueError):
            AgentEvent(type=AgentEventType.TOOL_EXECUTION_UPDATE, progress=-1)

    def test_event_with_stats(self):
        """测试带统计信息的事件."""
        stats = {"tokens": 100, "time": 1.5}
        event = AgentEvent(
            type=AgentEventType.STATS,
            stats=stats,
        )
        assert event.stats == stats

    def test_event_with_tool_results(self):
        """测试带工具结果的事件."""
        results = [{"tool": "time", "result": "2024-01-01"}]
        event = AgentEvent(
            type=AgentEventType.TURN_END,
            tool_results=results,
        )
        assert event.tool_results == results

    def test_event_repr(self):
        """测试事件的字符串表示."""
        event = AgentEvent(type=AgentEventType.AGENT_START)
        repr_str = repr(event)
        assert "AgentEvent" in repr_str
        assert "AGENT_START" in repr_str

    def test_event_str(self):
        """测试事件的人类可读表示."""
        event = AgentEvent(type=AgentEventType.AGENT_START)
        str_repr = str(event)
        assert "AgentEvent" in str_repr

    def test_event_with_tool_name_in_repr(self):
        """测试带工具名时的 repr."""
        event = AgentEvent(
            type=AgentEventType.TOOL_EXECUTION_START,
            tool_name="test_tool",
        )
        repr_str = repr(event)
        assert "test_tool" in repr_str

    def test_event_with_error_in_repr(self):
        """测试带错误时的 repr."""
        event = AgentEvent(
            type=AgentEventType.ERROR,
            error="test error",
        )
        repr_str = repr(event)
        assert "test error" in repr_str

    def test_event_with_progress_in_repr(self):
        """测试带进度时的 repr."""
        event = AgentEvent(
            type=AgentEventType.TOOL_EXECUTION_UPDATE,
            progress=75,
        )
        repr_str = repr(event)
        assert "75%" in repr_str

    def test_event_extra_fields_forbidden(self):
        """测试禁止额外字段."""
        with pytest.raises(ValueError):
            AgentEvent(
                type=AgentEventType.AGENT_START,
                unknown_field="value",
            )

    def test_event_mutable(self):
        """测试事件是可变的（方便修改）."""
        event = AgentEvent(type=AgentEventType.AGENT_START)
        event.message = "updated"
        assert event.message == "updated"


class TestChatResult:
    """测试 ChatResult 模型."""

    def test_basic_result_creation(self):
        """测试创建基本结果."""
        result = ChatResult(content="Hello!")
        assert result.content == "Hello!"
        assert result.messages == []
        assert result.usage == {}

    def test_full_result_creation(self):
        """测试创建完整结果."""
        result = ChatResult(
            content="Response",
            messages=[{"role": "assistant", "content": "Response"}],
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            model="gpt-4",
            finish_reason="stop",
        )
        assert result.content == "Response"
        assert len(result.messages) == 1
        assert result.usage["prompt_tokens"] == 10
        assert result.model == "gpt-4"
        assert result.finish_reason == "stop"

    def test_result_is_immutable(self):
        """测试结果是不可变的."""
        result = ChatResult(content="test")
        with pytest.raises(Exception):  # frozen model
            result.content = "modified"


class TestTokenStats:
    """测试 TokenStats 统计类."""

    def test_basic_stats_creation(self):
        """测试创建基本统计."""
        stats = TokenStats()
        assert stats.tokens == 0
        assert stats.r_tokens == 0
        assert stats.c_tokens == 0

    def test_stats_with_values(self):
        """测试带值的统计."""
        stats = TokenStats(tokens=100, r_tokens=30, c_tokens=70)
        assert stats.tokens == 100
        assert stats.r_tokens == 30
        assert stats.c_tokens == 70

    def test_stats_with_timestamps(self):
        """测试带时间戳的统计."""
        import time

        start = time.time()
        stats = TokenStats(start_time=start)
        assert stats.start_time == start

    def test_elapsed_property(self):
        """测试 elapsed 属性."""
        import time

        start = time.time() - 5  # 5秒前
        stats = TokenStats(start_time=start)
        stats.finalize()  # 需要 finalize 才能计算 elapsed
        assert stats.elapsed >= 4.9  # 允许误差

    def test_elapsed_zero_when_no_end(self):
        """测试没有结束时间时的 elapsed."""
        stats = TokenStats(start_time=100.0)
        # end_time 默认为 0.0
        assert stats.elapsed == 0.0

    def test_tps_property(self):
        """测试 TPS (tokens per second) 属性."""
        import time

        start = time.time() - 10  # 10秒前
        stats = TokenStats(tokens=100, start_time=start)
        stats.finalize()  # 需要 finalize 才能计算 TPS
        assert stats.tps >= 9.0  # 100/10 = 10 TPS, 允许误差

    def test_tps_zero_when_no_elapsed(self):
        """测试没有耗时时的 TPS."""
        stats = TokenStats(tokens=100)
        assert stats.tps == 0.0

    def test_ttft_property(self):
        """测试 TTFT (time to first token) 属性."""
        import time

        start = time.time() - 2
        first_token = time.time() - 1.5
        stats = TokenStats(start_time=start, first_token_time=first_token)
        assert stats.ttft >= 0.49  # 约 0.5 秒，允许误差

    def test_ttft_zero_when_no_first_token(self):
        """测试没有首 token 时的 TTFT."""
        stats = TokenStats(start_time=100.0)
        assert stats.ttft == 0.0

    def test_on_token_method(self):
        """测试 on_token 方法."""
        import time

        stats = TokenStats()
        stats.start_time = time.time() - 1  # 设置开始时间

        # 第一次调用应该设置 first_token_time
        stats.on_token()
        assert stats.first_token_time is not None

        # 第二次调用不应该改变 first_token_time
        first_time = stats.first_token_time
        stats.on_token()
        assert stats.first_token_time == first_time

    def test_finalize_method(self):
        """测试 finalize 方法."""
        stats = TokenStats()
        stats.finalize()
        assert stats.end_time is not None

    def test_str_representation_with_zero_tokens(self):
        """测试零 token 时的字符串表示."""
        stats = TokenStats()
        str_repr = str(stats)
        assert "0 TOKENS" in str_repr
        assert "0.0s" in str_repr
        assert "TPS 0.0" in str_repr

    def test_str_representation_with_tokens(self):
        """测试有 token 时的字符串表示."""
        import time

        start = time.time() - 5
        stats = TokenStats(
            tokens=100,
            r_tokens=30,
            c_tokens=70,
            start_time=start,
        )
        stats.finalize()
        str_repr = str(stats)
        assert "100 TOKENS" in str_repr
        assert "30.0% REASONING" in str_repr
        assert "70.0% CONTENT" in str_repr

    def test_token_count_validation(self):
        """测试 token 数必须非负."""
        with pytest.raises(ValueError):
            TokenStats(tokens=-1)

        with pytest.raises(ValueError):
            TokenStats(r_tokens=-1)

        with pytest.raises(ValueError):
            TokenStats(c_tokens=-1)


class TestBackwardCompatibility:
    """测试向后兼容的旧事件类型."""

    def test_old_event_types_exist(self):
        """测试旧的事件类型仍然存在."""
        assert hasattr(EventType, "REASONING_TOKEN")
        assert hasattr(EventType, "CONTENT_TOKEN")
        assert hasattr(EventType, "STATS")
        assert hasattr(EventType, "DONE")
        assert hasattr(EventType, "ERROR")

    def test_old_event_types_are_enum(self):
        """测试旧事件类型是枚举."""
        assert isinstance(EventType.REASONING_TOKEN, EventType)


class TestAgentEventExamples:
    """测试文档中的示例代码."""

    def test_doc_example_basic_event(self):
        """测试文档中的基本事件示例."""
        event = AgentEvent(type=AgentEventType.AGENT_START)
        assert event.type == AgentEventType.AGENT_START

    def test_doc_example_tool_event(self):
        """测试文档中的工具事件示例."""
        event = AgentEvent(
            type=AgentEventType.TOOL_EXECUTION_START,
            tool_call_id="call_123",
            tool_name="get_current_time",
            args={"timezone": "UTC"},
        )
        assert event.tool_call_id == "call_123"
        assert event.tool_name == "get_current_time"

    def test_doc_example_chat_result(self):
        """测试文档中的 ChatResult 示例."""
        result = ChatResult(
            content="Hello!",
            messages=[{"role": "assistant", "content": "Hello!"}],
            usage={"prompt_tokens": 10, "completion_tokens": 2},
            model="gpt-4",
        )
        assert result.content == "Hello!"

    def test_doc_example_token_stats(self):
        """测试文档中的 TokenStats 示例."""
        stats = TokenStats(tokens=100, r_tokens=30, c_tokens=70)
        tps_str = f"TPS: {stats.tps:.1f}"
        assert "TPS:" in tps_str
