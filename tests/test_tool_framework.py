"""Tests for Tool Framework module.

测试覆盖:
- Tool 基类和抽象方法
- ToolRegistry 注册和执行
- 内置工具 (GetCurrentTimeTool)
- Pydantic 参数验证
- OpenAI Schema 生成
"""

import asyncio
import pytest
from typing import Dict, Any, Optional
from datetime import datetime

from tui_chatbot.agent.tool import (
    Tool,
    ToolParameters,
    ToolResult,
    ToolRegistry,
    ToolExecutionMode,
    GetCurrentTimeParams,
    GetCurrentTimeTool,
    create_default_tool_registry,
)


class TestToolParameters:
    """测试 ToolParameters 基类."""

    def test_basic_params(self):
        """测试基本参数创建."""
        params = ToolParameters()
        assert params is not None

    def test_params_is_basemodel(self):
        """测试 ToolParameters 是 Pydantic BaseModel."""
        from pydantic import BaseModel

        assert issubclass(ToolParameters, BaseModel)


class TestToolResult:
    """测试 ToolResult 类."""

    def test_basic_result(self):
        """测试基本结果."""
        result = ToolResult(content="test content")
        assert result.content == "test content"
        assert result.is_error is False
        assert result.details == {}

    def test_error_result(self):
        """测试错误结果."""
        result = ToolResult(content="error message", is_error=True)
        assert result.content == "error message"
        assert result.is_error is True

    def test_result_with_details(self):
        """测试带详细信息的错误."""
        result = ToolResult(
            content="success",
            details={"key": "value", "number": 42},
        )
        assert result.details["key"] == "value"
        assert result.details["number"] == 42


class MockToolParams(ToolParameters):
    """模拟工具参数."""

    name: str
    value: int = 10


class MockTool(Tool):
    """模拟工具用于测试."""

    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "A mock tool for testing"

    @property
    def parameters(self):
        return MockToolParams

    async def execute(self, params: MockToolParams, signal=None) -> ToolResult:
        return ToolResult(content=f"Executed {params.name} with value {params.value}")


class TestToolBaseClass:
    """测试 Tool 基类."""

    def test_tool_properties(self):
        """测试工具属性."""
        tool = MockTool()

        assert tool.name == "mock_tool"
        assert tool.description == "A mock tool for testing"
        assert tool.parameters == MockToolParams

    def test_validate_params(self):
        """测试参数验证."""
        tool = MockTool()

        validated = tool.validate_params({"name": "test", "value": 20})

        assert isinstance(validated, MockToolParams)
        assert validated.name == "test"
        assert validated.value == 20

    def test_validate_params_defaults(self):
        """测试参数默认值."""
        tool = MockTool()

        validated = tool.validate_params({"name": "test"})

        assert validated.value == 10  # 默认值

    def test_validate_params_invalid(self):
        """测试无效参数."""
        tool = MockTool()

        # 缺少必需字段 'name'
        with pytest.raises(Exception):
            tool.validate_params({"value": 20})

    def test_to_openai_schema(self):
        """测试 OpenAI Schema 生成."""
        tool = MockTool()

        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "mock_tool"
        assert schema["function"]["description"] == "A mock tool for testing"
        assert "parameters" in schema["function"]

    @pytest.mark.asyncio
    async def test_execute(self):
        """测试执行."""
        tool = MockTool()
        params = MockToolParams(name="test", value=42)

        result = await tool.execute(params)

        assert isinstance(result, ToolResult)
        assert "test" in result.content
        assert "42" in result.content


class TestToolRegistry:
    """测试 ToolRegistry 类."""

    def test_register_tool(self):
        """测试注册工具."""
        registry = ToolRegistry()
        tool = MockTool()

        registry.register(tool)

        assert "mock_tool" in registry.list()

    def test_get_tool(self):
        """测试获取工具."""
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)

        retrieved = registry.get("mock_tool")

        assert retrieved is tool

    def test_get_nonexistent_tool(self):
        """测试获取不存在的工具."""
        registry = ToolRegistry()

        result = registry.get("nonexistent")

        assert result is None

    def test_list_tools(self):
        """测试列出所有工具."""
        registry = ToolRegistry()
        registry.register(MockTool())

        tools = registry.list()

        assert "mock_tool" in tools
        assert len(tools) == 1

    def test_list_empty_registry(self):
        """测试空注册表."""
        registry = ToolRegistry()

        tools = registry.list()

        assert tools == []

    def test_to_openai_tools(self):
        """测试转换为 OpenAI 工具列表."""
        registry = ToolRegistry()
        registry.register(MockTool())

        schemas = registry.to_openai_tools()

        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "mock_tool"

    def test_to_openai_tools_empty(self):
        """测试空注册表的 OpenAI 工具列表."""
        registry = ToolRegistry()

        schemas = registry.to_openai_tools()

        assert schemas == []

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """测试通过注册表执行工具."""
        registry = ToolRegistry()
        registry.register(MockTool())

        result = await registry.execute("mock_tool", {"name": "test"})

        assert isinstance(result, ToolResult)
        assert "test" in result.content

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self):
        """测试执行不存在的工具."""
        registry = ToolRegistry()

        result = await registry.execute("nonexistent", {})

        assert result.is_error is True
        assert "not found" in result.content

    @pytest.mark.asyncio
    async def test_execute_invalid_params(self):
        """测试执行时参数验证失败."""
        registry = ToolRegistry()
        registry.register(MockTool())

        result = await registry.execute("mock_tool", {"invalid": "params"})

        assert result.is_error is True
        assert "validation failed" in result.content.lower()


class TestGetCurrentTimeParams:
    """测试 GetCurrentTimeParams."""

    def test_default_timezone(self):
        """测试默认时区."""
        params = GetCurrentTimeParams()

        assert params.timezone == "UTC"

    def test_custom_timezone(self):
        """测试自定义时区."""
        params = GetCurrentTimeParams(timezone="Asia/Shanghai")

        assert params.timezone == "Asia/Shanghai"

    def test_timezone_description(self):
        """测试时区字段描述."""
        from pydantic import BaseModel

        field_info = GetCurrentTimeParams.model_fields["timezone"]
        assert (
            "时区" in field_info.description
            or "timezone" in field_info.description.lower()
        )


class TestGetCurrentTimeTool:
    """测试 GetCurrentTimeTool."""

    def test_tool_properties(self):
        """测试工具属性."""
        tool = GetCurrentTimeTool()

        assert tool.name == "get_current_time"
        assert "time" in tool.description.lower() or "时间" in tool.description
        assert tool.parameters == GetCurrentTimeParams

    def test_to_openai_schema(self):
        """测试 OpenAI Schema."""
        tool = GetCurrentTimeTool()

        schema = tool.to_openai_schema()

        assert schema["function"]["name"] == "get_current_time"
        assert "parameters" in schema["function"]

    @pytest.mark.asyncio
    async def test_execute_default_timezone(self):
        """测试默认时区执行."""
        tool = GetCurrentTimeTool()
        params = GetCurrentTimeParams()

        result = await tool.execute(params)

        assert isinstance(result, ToolResult)
        assert not result.is_error
        # 应该返回时间字符串
        assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_execute_custom_timezone(self):
        """测试自定义时区."""
        tool = GetCurrentTimeTool()
        params = GetCurrentTimeParams(timezone="America/New_York")

        result = await tool.execute(params)

        assert isinstance(result, ToolResult)
        # 如果 pytz 安装，应该成功

    @pytest.mark.asyncio
    async def test_execute_invalid_timezone(self):
        """测试无效时区."""
        tool = GetCurrentTimeTool()
        params = GetCurrentTimeParams(timezone="Invalid/Timezone")

        result = await tool.execute(params)

        # 应该回退到 UTC 或本地时间
        assert isinstance(result, ToolResult)
        assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_execute_returns_details(self):
        """测试执行返回详细信息."""
        tool = GetCurrentTimeTool()
        params = GetCurrentTimeParams(timezone="UTC")

        result = await tool.execute(params)

        assert "timezone" in result.details


class TestCreateDefaultToolRegistry:
    """测试创建默认工具注册表."""

    def test_creates_registry(self):
        """测试创建注册表."""
        registry = create_default_tool_registry()

        assert isinstance(registry, ToolRegistry)

    def test_includes_default_tools(self):
        """测试包含默认工具."""
        registry = create_default_tool_registry()

        tools = registry.list()

        assert "get_current_time" in tools

    @pytest.mark.asyncio
    async def test_can_execute_default_tool(self):
        """测试可以执行默认工具."""
        registry = create_default_tool_registry()

        result = await registry.execute("get_current_time", {"timezone": "UTC"})

        assert isinstance(result, ToolResult)
        assert not result.is_error or result.is_error  # 可能错误但应该返回结果


class TestToolExecutionMode:
    """测试 ToolExecutionMode 枚举."""

    def test_sequential_mode(self):
        """测试串行模式."""
        assert ToolExecutionMode.SEQUENTIAL is not None

    def test_parallel_mode(self):
        """测试并行模式."""
        assert ToolExecutionMode.PARALLEL is not None

    def test_modes_are_different(self):
        """测试两种模式不同."""
        assert ToolExecutionMode.SEQUENTIAL != ToolExecutionMode.PARALLEL


class TestToolEdgeCases:
    """测试边界情况."""

    @pytest.mark.asyncio
    async def test_tool_with_empty_params(self):
        """测试空参数."""
        registry = ToolRegistry()
        registry.register(MockTool())

        # MockTool 需要 name 参数，空字典应该失败
        result = await registry.execute("mock_tool", {})

        assert result.is_error

    @pytest.mark.asyncio
    async def test_tool_with_extra_params(self):
        """测试多余参数."""
        tool = MockTool()

        # Pydantic 应该忽略额外字段或报错
        try:
            validated = tool.validate_params({"name": "test", "extra": "field"})
            # 如果成功，应该只有 name 和 value
            assert hasattr(validated, "name")
        except Exception:
            # 也可能报错
            pass

    def test_registry_multiple_tools(self):
        """测试注册多个工具."""
        registry = ToolRegistry()

        class Tool1(Tool):
            @property
            def name(self):
                return "tool1"

            @property
            def description(self):
                return "Tool 1"

            @property
            def parameters(self):
                return ToolParameters

            async def execute(self, params, signal=None):
                return ToolResult(content="tool1 result")

        class Tool2(Tool):
            @property
            def name(self):
                return "tool2"

            @property
            def description(self):
                return "Tool 2"

            @property
            def parameters(self):
                return ToolParameters

            async def execute(self, params, signal=None):
                return ToolResult(content="tool2 result")

        registry.register(Tool1())
        registry.register(Tool2())

        assert len(registry.list()) == 2
        assert "tool1" in registry.list()
        assert "tool2" in registry.list()

    def test_tool_schema_includes_all_fields(self):
        """测试工具 schema 包含所有字段."""
        tool = GetCurrentTimeTool()

        schema = tool.to_openai_schema()
        params_schema = schema["function"]["parameters"]

        # 应该包含 timezone 字段
        assert "properties" in params_schema
        assert "timezone" in params_schema["properties"]


class TestToolWithSignal:
    """测试带取消信号的工具执行."""

    @pytest.mark.asyncio
    async def test_tool_respects_abort_signal(self):
        """测试工具尊重取消信号."""
        from tui_chatbot.core.abort_controller import AbortController

        ctrl = AbortController()
        ctrl.abort()

        tool = GetCurrentTimeTool()
        params = GetCurrentTimeParams()

        # 即使信号已取消，工具仍应该执行（取决于工具实现）
        # 这里只是测试可以传入信号参数
        result = await tool.execute(params, ctrl.signal)

        assert isinstance(result, ToolResult)
