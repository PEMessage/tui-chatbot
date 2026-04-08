"""工具框架 - Tool Registry 和内置工具."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Callable, Awaitable
from datetime import datetime
from enum import Enum, auto

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# Tool Configuration
# ═══════════════════════════════════════════════════════════════


class ToolExecutionMode(Enum):
    """工具执行模式."""

    SEQUENTIAL = auto()
    PARALLEL = auto()


# ═══════════════════════════════════════════════════════════════
# Tool Parameters & Results
# ═══════════════════════════════════════════════════════════════


class ToolParameters(BaseModel):
    """工具参数基类 - Pydantic 自动验证."""

    pass


class ToolResult(BaseModel):
    """工具执行结果."""

    content: str
    is_error: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# Tool Base Class
# ═══════════════════════════════════════════════════════════════


class Tool(ABC):
    """抽象工具基类."""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述 (用于 LLM)."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Type[ToolParameters]:
        """参数模型类."""
        pass

    @abstractmethod
    async def execute(
        self, params: ToolParameters, signal: Optional[Any] = None
    ) -> ToolResult:
        """执行工具."""
        pass

    def validate_params(self, args: Dict[str, Any]) -> ToolParameters:
        """验证参数 (使用 Pydantic)."""
        return self.parameters(**args)

    def to_openai_schema(self) -> Dict[str, Any]:
        """转换为 OpenAI 工具格式."""
        schema = self.parameters.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


# ═══════════════════════════════════════════════════════════════
# Tool Registry
# ═══════════════════════════════════════════════════════════════


class ToolRegistry:
    """工具注册表."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册工具."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """获取工具."""
        return self._tools.get(name)

    def list(self) -> List[str]:
        """列出所有工具名称."""
        return list(self._tools.keys())

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """转换为 OpenAI 工具列表."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    async def execute(
        self, name: str, args: Dict[str, Any], signal: Optional[Any] = None
    ) -> ToolResult:
        """执行指定工具."""
        tool = self.get(name)
        if not tool:
            return ToolResult(content=f"Tool {name} not found", is_error=True)

        try:
            validated = tool.validate_params(args)
            return await tool.execute(validated, signal)
        except Exception as e:
            return ToolResult(
                content=f"Parameter validation failed: {e}", is_error=True
            )


# ═══════════════════════════════════════════════════════════════
# Built-in Tools
# ═══════════════════════════════════════════════════════════════


class GetCurrentTimeParams(ToolParameters):
    """获取当前时间参数."""

    timezone: Optional[str] = Field(
        default="UTC", description="时区，如 'UTC', 'Asia/Shanghai', 'America/New_York'"
    )


class GetCurrentTimeTool(Tool):
    """获取当前时间工具 - 带 pytz 时区支持."""

    @property
    def name(self) -> str:
        return "get_current_time"

    @property
    def description(self) -> str:
        return "获取当前日期和时间，支持指定时区"

    @property
    def parameters(self) -> Type[ToolParameters]:
        return GetCurrentTimeParams

    async def execute(
        self, params: GetCurrentTimeParams, signal: Optional[Any] = None
    ) -> ToolResult:
        """执行获取时间工具."""
        try:
            import pytz
        except ImportError:
            # Fallback 如果 pytz 未安装
            from datetime import datetime as dt

            now = dt.now()
            return ToolResult(
                content=now.strftime(
                    "%Y-%m-%d %H:%M:%S (local time, pytz not installed)"
                ),
                details={"timezone": params.timezone, "note": "pytz not available"},
            )

        from datetime import datetime as dt

        try:
            tz = pytz.timezone(params.timezone)
            now = dt.now(tz)
            return ToolResult(
                content=now.strftime("%Y-%m-%d %H:%M:%S %Z"),
                details={"timezone": params.timezone},
            )
        except Exception as e:
            # 时区无效，回退到 UTC
            try:
                tz = pytz.UTC
                now = dt.now(tz)
                return ToolResult(
                    content=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    is_error=True,
                    details={
                        "error": str(e),
                        "requested_timezone": params.timezone,
                        "fallback": "UTC",
                    },
                )
            except Exception:
                # 如果连 UTC 都失败，使用本地时间
                now = dt.now()
                return ToolResult(
                    content=now.strftime("%Y-%m-%d %H:%M:%S (local time)"),
                    is_error=True,
                    details={
                        "error": f"Failed to get timezone {params.timezone}: {e}",
                        "fallback": "local time",
                    },
                )


# ═══════════════════════════════════════════════════════════════
# Factory Function
# ═══════════════════════════════════════════════════════════════


def create_default_tool_registry() -> ToolRegistry:
    """创建包含默认工具的注册表."""
    registry = ToolRegistry()
    registry.register(GetCurrentTimeTool())
    return registry
