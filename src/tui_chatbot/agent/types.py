"""Agent Loop types.

Defines dataclasses and types for the agent conversation loop including:
- Agent messages and tools
- Agent context and configuration
- Agent lifecycle events
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional, Union

from ..types import AssistantMessage, ToolCall, ToolResultMessage, UserMessage


# ╭────────────────────────────────────────────────────────────╮
# │  Agent Messages                                              │
# ╰────────────────────────────────────────────────────────────╯

AgentMessage = Union[UserMessage, AssistantMessage, ToolResultMessage]


# ╭────────────────────────────────────────────────────────────╮
# │  Agent Tool                                                  │
# ╰────────────────────────────────────────────────────────────╯


@dataclass
class AgentTool:
    """Agent tool definition.

    A tool that can be invoked by the agent during conversation.
    The execute function is called with the tool arguments and should
    return the result as a string or dict.

    Example:
        async def get_weather(args: dict) -> str:
            city = args.get("city", "Unknown")
            return f"Weather in {city}: 72°F, sunny"

        tool = AgentTool(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            },
            execute=get_weather,
        )
    """

    name: str
    description: str
    parameters: dict  # JSON schema
    execute: Callable[[dict], Awaitable[Union[str, dict]]]


# ╭────────────────────────────────────────────────────────────╮
# │  Agent Context                                               │
# ╰────────────────────────────────────────────────────────────╯


@dataclass
class AgentContext:
    """Agent conversation context.

    Holds the state for an agent conversation including system prompt,
    message history, and available tools.

    Example:
        context = AgentContext(
            system_prompt="You are a helpful assistant.",
            messages=[UserMessage(content="Hello")],
            tools=[calculator_tool, datetime_tool],
        )
    """

    system_prompt: Optional[str] = None
    messages: List[AgentMessage] = field(default_factory=list)
    tools: List[AgentTool] = field(default_factory=list)


# ╭────────────────────────────────────────────────────────────╮
# │  Agent Loop Config                                           │
# ╰────────────────────────────────────────────────────────────╯


@dataclass
class AgentLoopConfig:
    """Configuration for the agent conversation loop.

        Controls limits, tool execution mode, and provides hooks for
    customizing agent behavior.

        Example:
            config = AgentLoopConfig(
                max_turns=20,
                tool_mode="parallel",
                before_tool_call=validate_tool_args,
                after_tool_call=log_tool_result,
            )
    """

    max_turns: int = 10
    max_tool_calls: int = 10
    tool_mode: str = "sequential"  # "sequential" or "parallel"
    convert_to_llm: Optional[Callable[[List[AgentMessage]], List[dict]]] = None
    transform_context: Optional[Callable[[AgentContext], AgentContext]] = None
    before_tool_call: Optional[Callable[[ToolCall], Awaitable[Optional[str]]]] = None
    after_tool_call: Optional[Callable[[ToolCall, Any], Awaitable[None]]] = None


# ╭────────────────────────────────────────────────────────────╮
# │  Agent Events                                                │
# ╰────────────────────────────────────────────────────────────╯


@dataclass
class AgentStartEvent:
    """Agent loop start event.

    Emitted at the beginning of the agent conversation loop.
    """

    type: str = "agent_start"
    context: AgentContext = field(default_factory=AgentContext)


@dataclass
class AgentEndEvent:
    """Agent loop end event.

    Emitted at the end of the agent conversation loop with all messages.
    """

    type: str = "agent_end"
    messages: List[AgentMessage] = field(default_factory=list)


@dataclass
class TurnStartEvent:
    """Turn start event.

    Emitted at the beginning of each conversation turn.
    """

    type: str = "turn_start"
    turn: int = 0


@dataclass
class TurnEndEvent:
    """Turn end event.

    Emitted at the end of each conversation turn.
    """

    type: str = "turn_end"
    turn: int = 0
    message: Optional[AssistantMessage] = None


@dataclass
class ToolExecutionStartEvent:
    """Tool execution start event.

    Emitted when a tool is about to be executed.
    """

    type: str = "tool_execution_start"
    tool_call: ToolCall = field(default_factory=ToolCall)


@dataclass
class ToolExecutionEndEvent:
    """Tool execution end event.

    Emitted when a tool execution completes (success or error).
    """

    type: str = "tool_execution_end"
    tool_call: ToolCall = field(default_factory=ToolCall)
    result: Any = None
    error: Optional[str] = None


# Union type for all agent events
AgentEvent = Union[
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionEndEvent,
]
