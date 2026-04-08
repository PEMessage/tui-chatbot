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
# Calculator Tool
# ═══════════════════════════════════════════════════════════════


class CalculatorParams(ToolParameters):
    """计算器参数."""

    expression: str = Field(
        description="要计算的数学表达式，例如 '2 + 2' 或 'sin(pi/2)'"
    )


class CalculatorTool(Tool):
    """数学表达式计算工具.

    支持基本数学运算、科学计算函数。
    """

    @property
    def name(self) -> str:
        return "calculate"

    @property
    def description(self) -> str:
        return "计算数学表达式的结果，支持 +, -, *, /, **, sin, cos, log, sqrt 等运算"

    @property
    def parameters(self) -> Type[ToolParameters]:
        return CalculatorParams

    async def execute(
        self, params: CalculatorParams, signal: Optional[Any] = None
    ) -> ToolResult:
        """执行计算."""
        try:
            # 安全计算：限制可用函数
            allowed_names = {
                "abs": abs,
                "max": max,
                "min": min,
                "sum": sum,
                "pow": pow,
                "round": round,
                "sin": lambda x: __import__("math").sin(x),
                "cos": lambda x: __import__("math").cos(x),
                "tan": lambda x: __import__("math").tan(x),
                "sqrt": lambda x: __import__("math").sqrt(x),
                "log": lambda x: __import__("math").log(x),
                "log10": lambda x: __import__("math").log10(x),
                "exp": lambda x: __import__("math").exp(x),
                "pi": __import__("math").pi,
                "e": __import__("math").e,
            }

            # 计算表达式
            result = eval(params.expression, {"__builtins__": {}}, allowed_names)

            return ToolResult(
                content=f"{params.expression} = {result}",
                details={"expression": params.expression, "result": result},
            )
        except Exception as e:
            return ToolResult(
                content=f"计算错误: {e}",
                is_error=True,
                details={"error": str(e)},
            )


# ═══════════════════════════════════════════════════════════════
# File Tools
# ═══════════════════════════════════════════════════════════════


class ReadFileParams(ToolParameters):
    """读取文件参数."""

    file_path: str = Field(description="要读取的文件路径")
    limit: Optional[int] = Field(default=1000, description="最大读取行数，默认 1000")


class ReadFileTool(Tool):
    """文件读取工具."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "读取指定文件的内容，支持文本文件"

    @property
    def parameters(self) -> Type[ToolParameters]:
        return ReadFileParams

    async def execute(
        self, params: ReadFileParams, signal: Optional[Any] = None
    ) -> ToolResult:
        """执行文件读取."""
        try:
            from pathlib import Path

            path = Path(params.file_path).expanduser()

            if not path.exists():
                return ToolResult(
                    content=f"文件不存在: {params.file_path}", is_error=True
                )

            if not path.is_file():
                return ToolResult(
                    content=f"路径不是文件: {params.file_path}", is_error=True
                )

            # 读取文件
            content = path.read_text(encoding="utf-8", errors="replace")

            # 限制行数
            lines = content.split("\n")
            if len(lines) > params.limit:
                lines = lines[: params.limit]
                lines.append(f"\n... (已截断，仅显示前 {params.limit} 行)")
                content = "\n".join(lines)

            return ToolResult(
                content=content,
                details={
                    "file_path": str(path),
                    "size": path.stat().st_size,
                    "lines": len(lines),
                },
            )
        except Exception as e:
            return ToolResult(
                content=f"读取文件失败: {e}",
                is_error=True,
                details={"error": str(e)},
            )


class WriteFileParams(ToolParameters):
    """写入文件参数."""

    file_path: str = Field(description="要写入的文件路径")
    content: str = Field(description="文件内容")
    append: bool = Field(default=False, description="是否追加模式，默认覆盖")


class WriteFileTool(Tool):
    """文件写入工具."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "将内容写入指定文件，支持创建新文件或覆盖/追加现有文件"

    @property
    def parameters(self) -> Type[ToolParameters]:
        return WriteFileParams

    async def execute(
        self, params: WriteFileParams, signal: Optional[Any] = None
    ) -> ToolResult:
        """执行文件写入."""
        try:
            from pathlib import Path

            path = Path(params.file_path).expanduser()

            # 确保父目录存在
            path.parent.mkdir(parents=True, exist_ok=True)

            # 写入文件
            mode = "a" if params.append else "w"
            with open(path, mode, encoding="utf-8") as f:
                f.write(params.content)

            return ToolResult(
                content=f"文件已{'追加到' if params.append else '写入'}: {path}",
                details={
                    "file_path": str(path),
                    "size": path.stat().st_size,
                    "mode": "append" if params.append else "write",
                },
            )
        except Exception as e:
            return ToolResult(
                content=f"写入文件失败: {e}",
                is_error=True,
                details={"error": str(e)},
            )


# ═══════════════════════════════════════════════════════════════
# Factory Function
# ═══════════════════════════════════════════════════════════════


def create_default_tool_registry() -> ToolRegistry:
    """创建包含默认工具的注册表."""
    registry = ToolRegistry()
    registry.register(GetCurrentTimeTool())
    registry.register(CalculatorTool())
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    return registry
