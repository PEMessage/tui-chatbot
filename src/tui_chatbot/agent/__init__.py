"""Agent Loop modules.

Provides agent conversation loop with lifecycle events, tool execution,
and configuration hooks for extensible agent behavior.
"""

from .loop import agent_loop, AgentEventStream
from .tools import echo_tool, calculator_tool, datetime_tool
from .types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentStartEvent,
    AgentEndEvent,
    AgentTool,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)

__all__ = [
    # Main loop
    "agent_loop",
    "AgentEventStream",
    # Agent types
    "AgentMessage",
    "AgentTool",
    "AgentContext",
    "AgentLoopConfig",
    # Agent events
    "AgentEvent",
    "AgentStartEvent",
    "AgentEndEvent",
    "TurnStartEvent",
    "TurnEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionEndEvent",
    # Built-in tools
    "echo_tool",
    "calculator_tool",
    "datetime_tool",
]
