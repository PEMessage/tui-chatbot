"""Agent 模块 - 双层循环架构和工具框架."""

from .types import (
    AgentEvent,
    AgentEventType,
    AgentLoopConfig,
    AgentMessage,
    AssistantMessage,
    ChatResult,
    TextContent,
    ToolCallContent,
    ToolCallMessage,
    ToolExecutionMode,
    ToolResultMessage,
    UserMessage,
)
from .tool import (
    Tool,
    ToolParameters,
    ToolRegistry,
    ToolResult,
    GetCurrentTimeTool,
    GetCurrentTimeParams,
    create_default_tool_registry,
)
from .loop import (
    agent_loop,
    AgentEventSink,
)

__all__ = [
    # Types
    "AgentEvent",
    "AgentEventType",
    "AgentLoopConfig",
    "AgentMessage",
    "AssistantMessage",
    "ChatResult",
    "TextContent",
    "ToolCallContent",
    "ToolCallMessage",
    "ToolExecutionMode",
    "ToolResultMessage",
    "UserMessage",
    # Tool
    "Tool",
    "ToolParameters",
    "ToolRegistry",
    "ToolResult",
    "GetCurrentTimeTool",
    "GetCurrentTimeParams",
    "create_default_tool_registry",
    # Loop
    "agent_loop",
    "AgentEventSink",
]
