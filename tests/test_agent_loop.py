"""Agent Loop 集成测试.

测试覆盖:
- 基本对话流程
- 带工具调用的对话
- 多工具并行调用
- 错误处理

使用 MockProvider 避免依赖真实 API。
"""

import asyncio
import pytest
from typing import Any, Dict, List, Optional

from tui_chatbot.agent.loop import agent_loop
from tui_chatbot.agent.types import (
    AgentLoopConfig,
    AssistantMessage,
    TextContent,
    ToolCallContent,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
    ToolExecutionMode,
)
from tui_chatbot.agent.tool import (
    Tool,
    ToolParameters,
    ToolRegistry,
    ToolResult,
)
from tui_chatbot.core.events import AgentEvent, AgentEventType, ChatResult
from tui_chatbot.core.event_stream import EventStream
from tui_chatbot.provider.base import Provider


# ═════════════════════════════════════════════════════════════════════════════
# Mock Provider 实现
# ═════════════════════════════════════════════════════════════════════════════


class MockProvider(Provider):
    """用于测试的 Mock Provider.

    可以预设响应内容，包括:
    - 纯文本响应
    - 工具调用响应
    - 多工具调用响应
    - 错误响应
    """

    def __init__(
        self,
        responses: List[Any],
        name: str = "mock",
        api_type: str = "mock-chat",
    ):
        """初始化 Mock Provider.

        Args:
            responses: 响应列表，每个元素可以是:
                - str: 纯文本响应
                - List[ToolCallContent]: 工具调用列表
                - Exception: 要抛出的错误
            name: Provider 名称
            api_type: API 类型
        """
        self._responses = responses
        self._call_count = 0
        self._name = name
        self._api_type = api_type
        self.last_messages: List[dict] = []
        self.last_tools: Optional[List[dict]] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def api_type(self) -> str:
        return self._api_type

    async def stream_chat(
        self,
        model: str,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        signal: Optional[Any] = None,
    ) -> EventStream[AgentEvent, ChatResult]:
        """模拟流式对话.

        根据预设的响应生成事件流。
        """
        stream = EventStream[AgentEvent, ChatResult]()
        self.last_messages = messages
        self.last_tools = tools

        # 检查是否还有响应
        if self._call_count >= len(self._responses):
            # 默认返回空响应
            stream.end(ChatResult(content="", messages=[]))
            return stream

        response = self._responses[self._call_count]
        self._call_count += 1

        # 处理错误响应
        if isinstance(response, Exception):
            stream.error(response)
            return stream

        # 启动后台任务发送事件
        async def _send_events():
            try:
                # 发送消息开始事件
                assistant_msg = AssistantMessage(content=[], stop_reason="stop")
                stream.push(
                    AgentEvent(type=AgentEventType.MESSAGE_START, message=assistant_msg)
                )

                content_parts = []

                if isinstance(response, str):
                    # 纯文本响应
                    content_parts.append(TextContent(text=response))
                    stream.push(
                        AgentEvent(
                            type=AgentEventType.MESSAGE_UPDATE,
                            partial_result={"type": "content", "content": response},
                        )
                    )

                elif isinstance(response, list):
                    # 工具调用响应
                    for tool_call in response:
                        if isinstance(tool_call, ToolCallContent):
                            content_parts.append(tool_call)

                # 构建最终结果
                assistant_msg = AssistantMessage(
                    content=content_parts, stop_reason="stop"
                )

                # 发送消息结束事件
                stream.push(
                    AgentEvent(
                        type=AgentEventType.MESSAGE_END,
                        message=assistant_msg,
                    )
                )

                # 发送 TURN_END 事件
                stream.push(
                    AgentEvent(
                        type=AgentEventType.TURN_END,
                        message=assistant_msg,
                    )
                )

                # 设置最终结果
                result_messages = [{"role": "assistant", "content": content_parts}]
                stream.end(
                    ChatResult(
                        content="",
                        messages=result_messages,
                        model=model,
                        finish_reason="stop",
                    )
                )

            except Exception as e:
                stream.error(e)

        asyncio.create_task(_send_events())
        return stream

    async def list_models(self) -> List[str]:
        """返回模拟的模型列表."""
        return ["mock-model-1", "mock-model-2"]

    @property
    def call_count(self) -> int:
        """返回调用次数."""
        return self._call_count


# ═════════════════════════════════════════════════════════════════════════════
# Mock Tool 实现
# ═════════════════════════════════════════════════════════════════════════════


class MockToolParams(ToolParameters):
    """Mock 工具参数."""

    name: str = "default"
    value: int = 10


class MockEchoTool(Tool):
    """简单的 Echo 工具，用于测试."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo the input back"

    @property
    def parameters(self):
        return MockToolParams

    async def execute(self, params: MockToolParams, signal=None) -> ToolResult:
        return ToolResult(content=f"Echo: {params.name} = {params.value}")


class MockSlowTool(Tool):
    """模拟耗时工具，用于测试并行执行."""

    def __init__(self, delay: float = 0.1, tool_name: str = "slow_tool"):
        self._delay = delay
        self._tool_name = tool_name

    @property
    def name(self) -> str:
        return self._tool_name

    @property
    def description(self) -> str:
        return f"A slow tool with {self._delay}s delay"

    @property
    def parameters(self):
        return MockToolParams

    async def execute(self, params: MockToolParams, signal=None) -> ToolResult:
        await asyncio.sleep(self._delay)
        return ToolResult(
            content=f"Slow result for {params.name}",
            details={"delay": self._delay},
        )


class MockErrorTool(Tool):
    """模拟执行失败的工具."""

    @property
    def name(self) -> str:
        return "error_tool"

    @property
    def description(self) -> str:
        return "A tool that always fails"

    @property
    def parameters(self):
        return MockToolParams

    async def execute(self, params: MockToolParams, signal=None) -> ToolResult:
        raise RuntimeError(f"Tool execution failed for {params.name}")


class MockParamValidationTool(Tool):
    """需要特定参数的工具."""

    class Params(ToolParameters):
        required_field: str
        optional_field: str = "default"

    @property
    def name(self) -> str:
        return "validation_tool"

    @property
    def description(self) -> str:
        return "Tool with parameter validation"

    @property
    def parameters(self):
        return self.Params

    async def execute(self, params: Any, signal=None) -> ToolResult:
        return ToolResult(content=f"Valid: {params.required_field}")


# ═════════════════════════════════════════════════════════════════════════════
# 测试类: 基本对话
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestAgentLoopBasic:
    """测试基本对话流程."""

    async def test_basic_conversation(self):
        """测试不带工具的简单对话."""
        provider = MockProvider(["Hello, how can I help you?"])
        config = AgentLoopConfig(
            model="test-model",
            system_prompt="You are helpful.",
            tool_registry=ToolRegistry(),
            provider=provider,
        )

        events = []

        async def emit(event: AgentEvent):
            events.append(event)

        stream = await agent_loop(
            prompt=UserMessage(content="Hi"),
            history=[],
            config=config,
            emit=emit,
        )

        result = await stream.result()

        # 验证结果
        assert result is not None
        assert len(result) > 0
        assert result[0].role == "user"
        assert result[0].content == "Hi"

        # 验证事件序列
        assert len(events) > 0
        assert events[0].type == AgentEventType.AGENT_START
        assert events[-1].type == AgentEventType.AGENT_END

        # 验证中间事件包含 TURN_START 和 TURN_END
        event_types = [e.type for e in events]
        assert AgentEventType.TURN_START in event_types
        assert AgentEventType.TURN_END in event_types

        # 验证消息事件
        message_starts = [e for e in events if e.type == AgentEventType.MESSAGE_START]
        message_ends = [e for e in events if e.type == AgentEventType.MESSAGE_END]
        assert len(message_starts) >= 1  # 至少 prompt 的消息
        assert len(message_ends) >= 1

    async def test_conversation_with_history(self):
        """测试带历史记录的对话."""
        provider = MockProvider(["I remember our previous conversation."])
        config = AgentLoopConfig(
            model="test-model",
            system_prompt="You are helpful.",
            tool_registry=ToolRegistry(),
            provider=provider,
        )

        history = [
            UserMessage(content="Previous question"),
            AssistantMessage(content=[TextContent(text="Previous answer")]),
        ]

        events = []

        async def emit(event: AgentEvent):
            events.append(event)

        stream = await agent_loop(
            prompt=UserMessage(content="Follow up"),
            history=history,
            config=config,
            emit=emit,
        )

        result = await stream.result()

        # 验证历史被传递到 Provider
        assert len(provider.last_messages) > 0
        # 应该有系统提示 + 历史消息 + 当前提示
        assert len(provider.last_messages) >= 3

    async def test_empty_response(self):
        """测试空响应处理."""
        provider = MockProvider([""])  # 空响应
        config = AgentLoopConfig(
            model="test-model",
            system_prompt="You are helpful.",
            tool_registry=ToolRegistry(),
            provider=provider,
        )

        stream = await agent_loop(
            prompt=UserMessage(content="Hi"),
            history=[],
            config=config,
        )

        result = await stream.result()
        assert result is not None

    async def test_system_prompt_passed(self):
        """测试系统提示被正确传递."""
        provider = MockProvider(["Response"])
        config = AgentLoopConfig(
            model="test-model",
            system_prompt="You are a helpful assistant.",
            tool_registry=ToolRegistry(),
            provider=provider,
        )

        stream = await agent_loop(
            prompt=UserMessage(content="Hi"),
            history=[],
            config=config,
        )

        await stream.result()

        # 验证系统提示被传递
        assert len(provider.last_messages) > 0
        assert provider.last_messages[0]["role"] == "system"
        assert provider.last_messages[0]["content"] == "You are a helpful assistant."


# ═════════════════════════════════════════════════════════════════════════════
# 测试类: 带工具对话
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestAgentLoopWithTools:
    """测试带工具调用的对话."""

    async def test_single_tool_call(self):
        """测试单个工具调用."""
        registry = ToolRegistry()
        registry.register(MockEchoTool())

        # 第一次响应：工具调用
        # 第二次响应：最终结果
        provider = MockProvider(
            [
                [
                    ToolCallContent(
                        id="call_1",
                        name="echo",
                        arguments={"name": "test", "value": 42},
                    )
                ],
                "Tool executed successfully.",
            ]
        )

        config = AgentLoopConfig(
            model="test-model",
            system_prompt="You are helpful.",
            tool_registry=registry,
            provider=provider,
        )

        events = []

        async def emit(event: AgentEvent):
            events.append(event)

        stream = await agent_loop(
            prompt=UserMessage(content="Call echo tool"),
            history=[],
            config=config,
            emit=emit,
        )

        result = await stream.result()

        # 验证 Provider 被调用了两次
        assert provider.call_count == 2

        # 验证工具执行事件
        tool_starts = [
            e for e in events if e.type == AgentEventType.TOOL_EXECUTION_START
        ]
        tool_ends = [e for e in events if e.type == AgentEventType.TOOL_EXECUTION_END]
        assert len(tool_starts) == 1
        assert len(tool_ends) == 1

        # 验证工具调用信息
        assert tool_starts[0].tool_name == "echo"
        assert tool_starts[0].args == {"name": "test", "value": 42}

        # 验证工具结果
        assert tool_ends[0].is_error is False
        assert "Echo: test = 42" in tool_ends[0].result

        # 验证结果包含工具结果消息
        tool_result_msgs = [m for m in result if isinstance(m, ToolResultMessage)]
        assert len(tool_result_msgs) == 1
        assert tool_result_msgs[0].content == "Echo: test = 42"

    async def test_tool_not_found(self):
        """测试工具不存在的情况."""
        registry = ToolRegistry()  # 空注册表

        provider = MockProvider(
            [
                [ToolCallContent(id="call_1", name="nonexistent_tool", arguments={})],
                "I couldn't find that tool.",
            ]
        )

        config = AgentLoopConfig(
            model="test-model",
            system_prompt="You are helpful.",
            tool_registry=registry,
            provider=provider,
        )

        events = []

        async def emit(event: AgentEvent):
            events.append(event)

        stream = await agent_loop(
            prompt=UserMessage(content="Call nonexistent tool"),
            history=[],
            config=config,
            emit=emit,
        )

        result = await stream.result()

        # 验证工具执行结束事件标记为错误
        tool_ends = [e for e in events if e.type == AgentEventType.TOOL_EXECUTION_END]
        assert len(tool_ends) == 1
        assert tool_ends[0].is_error is True
        assert "not found" in tool_ends[0].result

        # 验证结果包含错误消息
        tool_result_msgs = [m for m in result if isinstance(m, ToolResultMessage)]
        assert len(tool_result_msgs) == 1
        assert tool_result_msgs[0].is_error is True

    async def test_tools_schema_passed(self):
        """测试工具 schema 被传递给 Provider."""
        registry = ToolRegistry()
        registry.register(MockEchoTool())

        provider = MockProvider(["No tools needed."])
        config = AgentLoopConfig(
            model="test-model",
            system_prompt="You are helpful.",
            tool_registry=registry,
            provider=provider,
        )

        stream = await agent_loop(
            prompt=UserMessage(content="Hi"),
            history=[],
            config=config,
        )

        await stream.result()

        # 验证工具 schema 被传递
        assert provider.last_tools is not None
        assert len(provider.last_tools) == 1
        assert provider.last_tools[0]["function"]["name"] == "echo"


# ═════════════════════════════════════════════════════════════════════════════
# 测试类: 多工具调用
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestAgentLoopMultiTools:
    """测试多工具调用场景."""

    async def test_multiple_tools_parallel(self):
        """测试一次响应中调用多个工具（并行模式）."""
        registry = ToolRegistry()
        registry.register(MockSlowTool(delay=0.05, tool_name="slow_1"))
        registry.register(MockSlowTool(delay=0.05, tool_name="slow_2"))
        registry.register(MockSlowTool(delay=0.05, tool_name="slow_3"))

        provider = MockProvider(
            [
                [
                    ToolCallContent(
                        id="call_1", name="slow_1", arguments={"name": "first"}
                    ),
                    ToolCallContent(
                        id="call_2", name="slow_2", arguments={"name": "second"}
                    ),
                    ToolCallContent(
                        id="call_3", name="slow_3", arguments={"name": "third"}
                    ),
                ],
                "All tools executed.",
            ]
        )

        config = AgentLoopConfig(
            model="test-model",
            system_prompt="You are helpful.",
            tool_registry=registry,
            provider=provider,
            tool_execution_mode=ToolExecutionMode.PARALLEL,
        )

        events = []

        async def emit(event: AgentEvent):
            events.append(event)

        stream = await agent_loop(
            prompt=UserMessage(content="Call multiple tools"),
            history=[],
            config=config,
            emit=emit,
        )

        result = await stream.result()

        # 验证三个工具都被执行
        tool_starts = [
            e for e in events if e.type == AgentEventType.TOOL_EXECUTION_START
        ]
        tool_ends = [e for e in events if e.type == AgentEventType.TOOL_EXECUTION_END]
        assert len(tool_starts) == 3
        assert len(tool_ends) == 3

        # 验证工具结果消息
        tool_result_msgs = [m for m in result if isinstance(m, ToolResultMessage)]
        assert len(tool_result_msgs) == 3

    async def test_multiple_tools_sequential(self):
        """测试一次响应中调用多个工具（串行模式）."""
        registry = ToolRegistry()
        registry.register(MockSlowTool(delay=0.03, tool_name="seq_1"))
        registry.register(MockSlowTool(delay=0.03, tool_name="seq_2"))

        provider = MockProvider(
            [
                [
                    ToolCallContent(
                        id="call_1", name="seq_1", arguments={"name": "first"}
                    ),
                    ToolCallContent(
                        id="call_2", name="seq_2", arguments={"name": "second"}
                    ),
                ],
                "Sequential execution complete.",
            ]
        )

        config = AgentLoopConfig(
            model="test-model",
            system_prompt="You are helpful.",
            tool_registry=registry,
            provider=provider,
            tool_execution_mode=ToolExecutionMode.SEQUENTIAL,
        )

        events = []

        async def emit(event: AgentEvent):
            events.append(event)

        stream = await agent_loop(
            prompt=UserMessage(content="Call tools sequentially"),
            history=[],
            config=config,
            emit=emit,
        )

        result = await stream.result()

        # 验证两个工具都被执行
        tool_starts = [
            e for e in events if e.type == AgentEventType.TOOL_EXECUTION_START
        ]
        tool_ends = [e for e in events if e.type == AgentEventType.TOOL_EXECUTION_END]
        assert len(tool_starts) == 2
        assert len(tool_ends) == 2

        # 验证工具按原始顺序返回结果
        tool_result_msgs = [m for m in result if isinstance(m, ToolResultMessage)]
        assert len(tool_result_msgs) == 2
        assert tool_result_msgs[0].tool_name == "seq_1"
        assert tool_result_msgs[1].tool_name == "seq_2"


# ═════════════════════════════════════════════════════════════════════════════
# 测试类: 错误处理
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestAgentLoopErrors:
    """测试错误处理场景."""

    async def test_tool_execution_error(self):
        """测试工具执行错误处理."""
        registry = ToolRegistry()
        registry.register(MockErrorTool())

        provider = MockProvider(
            [
                [
                    ToolCallContent(
                        id="call_1", name="error_tool", arguments={"name": "test"}
                    )
                ],
                "Sorry, the tool failed.",
            ]
        )

        config = AgentLoopConfig(
            model="test-model",
            system_prompt="You are helpful.",
            tool_registry=registry,
            provider=provider,
        )

        events = []

        async def emit(event: AgentEvent):
            events.append(event)

        stream = await agent_loop(
            prompt=UserMessage(content="Call error tool"),
            history=[],
            config=config,
            emit=emit,
        )

        result = await stream.result()

        # 验证工具执行错误事件
        tool_ends = [e for e in events if e.type == AgentEventType.TOOL_EXECUTION_END]
        assert len(tool_ends) == 1
        assert tool_ends[0].is_error is True

        # 验证结果包含错误消息
        tool_result_msgs = [m for m in result if isinstance(m, ToolResultMessage)]
        assert len(tool_result_msgs) == 1
        assert tool_result_msgs[0].is_error is True
        assert "execution failed" in tool_result_msgs[0].content.lower()

    async def test_provider_not_available(self):
        """测试 Provider 不可用时抛出错误."""
        config = AgentLoopConfig(
            model="test-model",
            system_prompt="You are helpful.",
            tool_registry=ToolRegistry(),
            provider=None,  # 没有 Provider
        )

        with pytest.raises(RuntimeError) as exc_info:
            stream = await agent_loop(
                prompt=UserMessage(content="Hi"),
                history=[],
                config=config,
            )
            await stream.result()

        assert "No provider available" in str(exc_info.value)

    async def test_tool_param_validation_error(self):
        """测试工具参数验证错误."""
        registry = ToolRegistry()
        registry.register(MockParamValidationTool())

        # 调用时缺少必需参数
        provider = MockProvider(
            [
                [
                    ToolCallContent(
                        id="call_1",
                        name="validation_tool",
                        arguments={"optional_field": "value"},  # 缺少 required_field
                    )
                ],
                "Invalid parameters.",
            ]
        )

        config = AgentLoopConfig(
            model="test-model",
            system_prompt="You are helpful.",
            tool_registry=registry,
            provider=provider,
        )

        events = []

        async def emit(event: AgentEvent):
            events.append(event)

        stream = await agent_loop(
            prompt=UserMessage(content="Call with invalid params"),
            history=[],
            config=config,
            emit=emit,
        )

        result = await stream.result()

        # 验证参数验证错误
        tool_ends = [e for e in events if e.type == AgentEventType.TOOL_EXECUTION_END]
        assert len(tool_ends) == 1
        assert tool_ends[0].is_error is True
        assert "validation failed" in tool_ends[0].result.lower()

    async def test_mixed_success_and_error_tools(self):
        """测试混合成功和失败的工具调用."""
        registry = ToolRegistry()
        registry.register(MockEchoTool())
        registry.register(MockErrorTool())

        provider = MockProvider(
            [
                [
                    ToolCallContent(
                        id="call_1", name="echo", arguments={"name": "success"}
                    ),
                    ToolCallContent(
                        id="call_2", name="error_tool", arguments={"name": "fail"}
                    ),
                ],
                "Mixed results.",
            ]
        )

        config = AgentLoopConfig(
            model="test-model",
            system_prompt="You are helpful.",
            tool_registry=registry,
            provider=provider,
        )

        events = []

        async def emit(event: AgentEvent):
            events.append(event)

        stream = await agent_loop(
            prompt=UserMessage(content="Call mixed tools"),
            history=[],
            config=config,
            emit=emit,
        )

        result = await stream.result()

        # 验证两个工具事件
        tool_ends = [e for e in events if e.type == AgentEventType.TOOL_EXECUTION_END]
        assert len(tool_ends) == 2

        # 一个成功，一个失败
        assert tool_ends[0].is_error is False  # echo
        assert tool_ends[1].is_error is True  # error_tool

        # 验证结果消息
        tool_result_msgs = [m for m in result if isinstance(m, ToolResultMessage)]
        assert len(tool_result_msgs) == 2
        assert tool_result_msgs[0].is_error is False
        assert tool_result_msgs[1].is_error is True


# ═════════════════════════════════════════════════════════════════════════════
# 测试类: 流式事件
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestAgentLoopStreaming:
    """测试流式事件输出."""

    async def test_event_stream_iteration(self):
        """测试通过 async for 迭代事件流."""
        provider = MockProvider(["Hello there!"])
        config = AgentLoopConfig(
            model="test-model",
            system_prompt="You are helpful.",
            tool_registry=ToolRegistry(),
            provider=provider,
        )

        stream = await agent_loop(
            prompt=UserMessage(content="Hi"),
            history=[],
            config=config,
        )

        # 通过迭代获取事件
        events = []
        async for event in stream:
            events.append(event)

        # 验证事件序列
        assert len(events) > 0
        assert events[0].type == AgentEventType.AGENT_START
        assert events[-1].type == AgentEventType.AGENT_END

    async def test_event_callback(self):
        """测试事件回调函数."""
        provider = MockProvider(["Response"])
        config = AgentLoopConfig(
            model="test-model",
            system_prompt="You are helpful.",
            tool_registry=ToolRegistry(),
            provider=provider,
        )

        received_events = []

        async def emit(event: AgentEvent):
            received_events.append(event)

        stream = await agent_loop(
            prompt=UserMessage(content="Hi"),
            history=[],
            config=config,
            emit=emit,
        )

        # 同时通过回调和迭代获取事件
        async for _ in stream:
            pass

        result = await stream.result()

        # 回调应该收到事件
        assert len(received_events) > 0

        # 结果应该正确
        assert result is not None


# ═════════════════════════════════════════════════════════════════════════════
# 测试类: 配置选项
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestAgentLoopConfig:
    """测试配置选项."""

    async def test_max_iterations_limit(self):
        """测试最大迭代次数限制."""
        provider = MockProvider(["Response"])
        config = AgentLoopConfig(
            model="test-model",
            system_prompt="You are helpful.",
            tool_registry=ToolRegistry(),
            provider=provider,
            max_iterations=1,  # 限制为1次迭代
        )

        stream = await agent_loop(
            prompt=UserMessage(content="Hi"),
            history=[],
            config=config,
        )

        result = await stream.result()
        assert result is not None

    async def test_temperature_and_max_tokens(self):
        """测试温度和最大 token 配置."""
        provider = MockProvider(["Response"])
        config = AgentLoopConfig(
            model="test-model",
            system_prompt="You are helpful.",
            tool_registry=ToolRegistry(),
            provider=provider,
            temperature=0.7,
            max_tokens=100,
        )

        stream = await agent_loop(
            prompt=UserMessage(content="Hi"),
            history=[],
            config=config,
        )

        result = await stream.result()
        assert result is not None

        # 验证配置被保存
        assert config.temperature == 0.7
        assert config.max_tokens == 100


# ═════════════════════════════════════════════════════════════════════════════
# 边界情况测试
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
class TestAgentLoopEdgeCases:
    """测试边界情况."""

    async def test_empty_history(self):
        """测试空历史记录."""
        provider = MockProvider(["Hello!"])
        config = AgentLoopConfig(
            model="test-model",
            system_prompt="You are helpful.",
            tool_registry=ToolRegistry(),
            provider=provider,
        )

        stream = await agent_loop(
            prompt=UserMessage(content="Hi"),
            history=[],  # 空历史
            config=config,
        )

        result = await stream.result()
        assert len(result) == 2  # prompt + response

    async def test_no_emit_callback(self):
        """测试不提供 emit 回调."""
        provider = MockProvider(["Hello!"])
        config = AgentLoopConfig(
            model="test-model",
            system_prompt="You are helpful.",
            tool_registry=ToolRegistry(),
            provider=provider,
        )

        stream = await agent_loop(
            prompt=UserMessage(content="Hi"),
            history=[],
            config=config,
            emit=None,  # 不提供回调
        )

        result = await stream.result()
        assert result is not None

    async def test_no_tool_registry(self):
        """测试不提供工具注册表."""
        provider = MockProvider(["Hello!"])
        config = AgentLoopConfig(
            model="test-model",
            system_prompt="You are helpful.",
            tool_registry=None,  # 无工具
            provider=provider,
        )

        stream = await agent_loop(
            prompt=UserMessage(content="Hi"),
            history=[],
            config=config,
        )

        result = await stream.result()
        assert result is not None

        # 验证工具 schema 为 None
        assert provider.last_tools is None

    async def test_concurrent_conversations(self):
        """测试并发对话."""
        provider1 = MockProvider(["Response 1"])
        provider2 = MockProvider(["Response 2"])

        config1 = AgentLoopConfig(
            model="model-1",
            system_prompt="You are A.",
            tool_registry=ToolRegistry(),
            provider=provider1,
        )
        config2 = AgentLoopConfig(
            model="model-2",
            system_prompt="You are B.",
            tool_registry=ToolRegistry(),
            provider=provider2,
        )

        async def run_conversation(config, prompt_text):
            stream = await agent_loop(
                prompt=UserMessage(content=prompt_text),
                history=[],
                config=config,
            )
            return await stream.result()

        # 并发运行
        results = await asyncio.gather(
            run_conversation(config1, "Hi from 1"),
            run_conversation(config2, "Hi from 2"),
        )

        # 验证两个对话都成功
        assert len(results) == 2
        assert all(len(r) > 0 for r in results)
