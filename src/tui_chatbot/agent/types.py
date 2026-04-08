"""Agent 相关类型定义."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, Field

# 从 core.events 导入 AgentEventType 避免重复定义
from ..core.events import AgentEventType


# ═══════════════════════════════════════════════════════════════
# Message Types
# ═══════════════════════════════════════════════════════════════


class AgentMessage(ABC, BaseModel):
    """Agent 消息基类."""

    role: str
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = {"frozen": True}


class UserMessage(AgentMessage):
    """用户消息."""

    role: Literal["user"] = "user"
    content: str


class TextContent(BaseModel):
    """文本内容块."""

    type: Literal["text"] = "text"
    text: str


class ToolCallContent(BaseModel):
    """工具调用内容块."""

    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class AssistantMessage(AgentMessage):
    """助手消息."""

    role: Literal["assistant"] = "assistant"
    content: List[Union[TextContent, ToolCallContent]] = Field(default_factory=list)
    stop_reason: Optional[str] = None
    error_message: Optional[str] = None


class ToolCallMessage(AgentMessage):
    """工具调用消息 (从助手消息提取)."""

    role: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResultMessage(AgentMessage):
    """工具执行结果消息."""

    role: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: Optional[str] = None
    content: str
    is_error: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# Agent Events (从 core.events 导入)
# ═══════════════════════════════════════════════════════════════

from ..core.events import AgentEvent, ChatResult


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════


class ToolExecutionMode(Enum):
    """工具执行模式."""

    SEQUENTIAL = auto()
    PARALLEL = auto()


@dataclass
class AgentLoopConfig:
    """Agent 循环配置."""

    model: str
    system_prompt: str
    tool_registry: Any  # ToolRegistry - 避免循环导入
    tool_execution_mode: ToolExecutionMode = ToolExecutionMode.PARALLEL
    max_iterations: int = 10  # 防止无限循环
    api_key: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# Type Aliases
# ═══════════════════════════════════════════════════════════════

AgentEventSink = Any  # Callable[[AgentEvent], Awaitable[None]]
