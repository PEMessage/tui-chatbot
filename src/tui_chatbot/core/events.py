"""事件类型定义 - 支持完整 Agent 生命周期."""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentEventType(Enum):
    """
    Agent 事件类型 - 完整生命周期覆盖.

    包含 Agent、Turn、Message、Tool 四个层次的生命周期事件。
    """

    # ╭────────────────────────────────────────────────────────────╮
    # │  Agent 生命周期 (外层循环)                                   │
    # ╰────────────────────────────────────────────────────────────╯
    AGENT_START = auto()
    """Agent 开始处理请求."""

    AGENT_END = auto()
    """Agent 完成所有处理."""

    # ╭────────────────────────────────────────────────────────────╮
    # │  Turn 生命周期 (一轮 = 助手响应 + 可能的工具调用)              │
    # ╰────────────────────────────────────────────────────────────╯
    TURN_START = auto()
    """新一轮对话开始."""

    TURN_END = auto()
    """一轮对话结束 (包括工具调用完成)."""

    # ╭────────────────────────────────────────────────────────────╮
    # │  消息生命周期 (流式输出)                                     │
    # ╰────────────────────────────────────────────────────────────╯
    MESSAGE_START = auto()
    """消息开始生成."""

    MESSAGE_UPDATE = auto()
    """消息内容更新 (流式增量)."""

    MESSAGE_END = auto()
    """消息生成完成."""

    # ╭────────────────────────────────────────────────────────────╮
    # │  工具生命周期                                               │
    # ╰────────────────────────────────────────────────────────────╯
    TOOL_EXECUTION_START = auto()
    """工具执行开始."""

    TOOL_EXECUTION_UPDATE = auto()
    """工具执行进度更新 (用于长时间运行工具)."""

    TOOL_EXECUTION_END = auto()
    """工具执行完成."""

    # ╭────────────────────────────────────────────────────────────╮
    # │  其他事件                                                   │
    # ╰────────────────────────────────────────────────────────────╯
    ERROR = auto()
    """发生错误."""

    STATS = auto()
    """统计信息 (Token 使用、耗时等)."""

    REASONING_TOKEN = auto()
    """推理 Token (来自 reasoning_content)."""

    CONTENT_TOKEN = auto()
    """内容 Token (来自 content)."""


class AgentEvent(BaseModel):
    """
    Agent 事件 - Pydantic 模型.

    包含类型、消息、工具结果、错误等信息。

    Attributes:
        type: 事件类型
        message: 关联消息内容 (可选)
        tool_results: 工具执行结果列表 (可选)
        error: 错误信息 (可选)
        tool_call_id: 工具调用 ID (可选)
        tool_name: 工具名称 (可选)
        args: 工具参数 (可选)
        result: 工具执行结果 (可选)
        is_error: 是否出错 (可选)
        progress: 进度百分比 0-100 (可选)
        stats: 统计信息字典 (可选)
        data: 任意附加数据 (可选)
        partial_result: 流式更新部分结果 (可选)

    Examples:
        >>> event = AgentEvent(type=AgentEventType.AGENT_START)
        >>> event = AgentEvent(
        ...     type=AgentEventType.TOOL_EXECUTION_START,
        ...     tool_call_id="call_123",
        ...     tool_name="get_current_time",
        ...     args={"timezone": "UTC"}
        ... )
    """

    type: AgentEventType
    message: Optional[Any] = Field(
        default=None, description="关联消息 (AssistantMessage)"
    )
    messages: List[Any] = Field(default_factory=list, description="消息列表")
    tool_results: List[Dict[str, Any]] = Field(
        default_factory=list, description="工具执行结果列表"
    )
    error: Optional[str] = Field(default=None, description="错误信息")
    tool_call_id: Optional[str] = Field(default=None, description="工具调用 ID")
    tool_name: Optional[str] = Field(default=None, description="工具名称")
    args: Optional[Dict[str, Any]] = Field(default=None, description="工具参数")
    result: Optional[Any] = Field(default=None, description="工具执行结果")
    is_error: bool = Field(default=False, description="是否出错")
    progress: Optional[int] = Field(
        default=None, ge=0, le=100, description="进度百分比 0-100"
    )
    stats: Optional[Dict[str, Any]] = Field(default=None, description="统计信息")
    data: Optional[Any] = Field(default=None, description="任意附加数据")
    partial_result: Optional[Dict[str, Any]] = Field(
        default=None, description="流式更新部分结果"
    )

    model_config = {
        "frozen": False,  # 允许修改，方便事件处理
        "extra": "forbid",  # 禁止额外字段，防止拼写错误
    }

    def __repr__(self) -> str:
        """简洁的字符串表示."""
        base = f"AgentEvent({self.type.name}"
        if self.tool_name:
            base += f", tool={self.tool_name}"
        if self.error:
            base += f", error={self.error!r}"
        if self.progress is not None:
            base += f", progress={self.progress}%"
        return base + ")"

    def __str__(self) -> str:
        """人类可读的字符串表示."""
        return self.__repr__()


class ChatResult(BaseModel):
    """
    对话结果 - 包含完整响应信息.

    Attributes:
        content: 最终响应内容
        messages: 所有生成的消息
        usage: Token 使用统计
        model: 使用的模型
        finish_reason: 结束原因

    Examples:
        >>> result = ChatResult(
        ...     content="Hello!",
        ...     messages=[{"role": "assistant", "content": "Hello!"}],
        ...     usage={"prompt_tokens": 10, "completion_tokens": 2},
        ...     model="gpt-4"
        ... )
    """

    content: str = Field(default="", description="最终响应内容")
    messages: List[Any] = Field(
        default_factory=list, description="所有生成的消息 (包括助手和工具消息)"
    )
    usage: Dict[str, int] = Field(default_factory=dict, description="Token 使用统计")
    model: Optional[str] = Field(default=None, description="使用的模型")
    finish_reason: Optional[str] = Field(default=None, description="结束原因")

    model_config = {
        "frozen": True,  # 不可变模型，保证数据不变性
        "extra": "forbid",  # 禁止额外字段
    }


class TokenStats(BaseModel):
    """
    Token 统计信息.

    Attributes:
        tokens: 总 Token 数
        r_tokens: 推理 Token 数
        c_tokens: 内容 Token 数
        start_time: 开始时间戳
        first_token_time: 首个 Token 时间戳
        end_time: 结束时间戳

    Examples:
        >>> stats = TokenStats(tokens=100, r_tokens=30, c_tokens=70)
        >>> print(f"TPS: {stats.tps:.1f}")
    """

    tokens: int = Field(default=0, ge=0, description="总 Token 数")
    r_tokens: int = Field(default=0, ge=0, description="推理 Token 数")
    c_tokens: int = Field(default=0, ge=0, description="内容 Token 数")
    start_time: Optional[float] = Field(default=None, description="开始时间戳")
    first_token_time: Optional[float] = Field(
        default=None, description="首个 Token 时间戳"
    )
    end_time: Optional[float] = Field(default=None, description="结束时间戳")

    @property
    def elapsed(self) -> float:
        """计算耗时，如果未结束返回 0.0."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return max(0.0, self.end_time - self.start_time)

    @property
    def tps(self) -> float:
        """Token 每秒 (吞吐量)."""
        elapsed = self.elapsed
        return self.tokens / elapsed if elapsed > 0 else 0.0

    @property
    def ttft(self) -> float:
        """首个 Token 延迟 (秒)."""
        if self.first_token_time and self.start_time:
            return max(0.0, self.first_token_time - self.start_time)
        return 0.0

    def on_token(self) -> None:
        """记录首个 Token 时间."""
        import time

        if self.first_token_time is None:
            self.first_token_time = time.time()

    def finalize(self) -> None:
        """标记结束时间."""
        import time

        self.end_time = time.time()

    def __str__(self) -> str:
        """格式化统计信息."""
        if self.tokens == 0:
            return "[0 TOKENS | 0.0s | TPS 0.0]"
        r_pct = (self.r_tokens / self.tokens) * 100 if self.tokens > 0 else 0
        c_pct = (self.c_tokens / self.tokens) * 100 if self.tokens > 0 else 0
        return (
            f"[{self.tokens} TOKENS | {r_pct:.1f}% REASONING | {c_pct:.1f}% CONTENT | "
            f"{self.elapsed:.1f}s | TPS {self.tps:.1f} | TTFT {self.ttft:.2f}s]"
        )


# ╭────────────────────────────────────────────────────────────╮
# │  向后兼容 - 旧事件类型 (保留供过渡期使用)                      │
# ╰────────────────────────────────────────────────────────────╯


class EventType(Enum):
    """
    旧事件类型 - 保留向后兼容.

    建议在 Phase 2 中逐步迁移到 AgentEventType。
    """

    REASONING_TOKEN = auto()
    CONTENT_TOKEN = auto()
    STATS = auto()
    DONE = auto()
    ERROR = auto()


# 导出所有公共 API
__all__ = [
    # 事件类型
    "AgentEventType",
    "EventType",  # 向后兼容
    # 事件模型
    "AgentEvent",
    "ChatResult",
    "TokenStats",
]
