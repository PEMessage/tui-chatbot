"""核心基础设施模块."""

from __future__ import annotations

# EventStream - 双接口事件流
from .event_stream import EventStream

# 事件类型和模型
from .events import (
    AgentEventType,
    AgentEvent,
    ChatResult,
    TokenStats,
    # 向后兼容
    EventType,
)

# AbortController 模式
from .abort_controller import (
    AbortSignal,
    AbortController,
    AbortManager,
)

__all__ = [
    # EventStream
    "EventStream",
    # 事件类型
    "AgentEventType",
    "AgentEvent",
    "ChatResult",
    "TokenStats",
    "EventType",  # 向后兼容
    # AbortController
    "AbortSignal",
    "AbortController",
    "AbortManager",
]
