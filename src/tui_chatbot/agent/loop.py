"""Agent Loop implementation.

The agent loop orchestrates conversation flow between user and assistant,
handles tool execution, and emits lifecycle events for monitoring and control.

Example:
    # Simple conversation
    stream = agent_loop(
        messages=[UserMessage(content="Hello")],
        provider=openai_provider,
        model="gpt-4",
    )

    # Monitor events
    async for event in stream:
        print(f"Event: {event.type}")

    # Get final result
    messages = await stream.result()
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

from ..event_stream import EventStream
from ..events import DoneEvent, ErrorEvent, is_terminal_event
from ..types import (
    AssistantMessage,
    Content,
    StopReason,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from .types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)

if TYPE_CHECKING:
    from ..core.abort_controller import AbortSignal
    from ..providers.base import Provider


# ╭────────────────────────────────────────────────────────────╮
# │  Agent Event Stream                                          │
# ╰────────────────────────────────────────────────────────────╯


class AgentEventStream(EventStream[AgentEvent, List[AgentMessage]]):
    """EventStream for Agent events.

    Terminal events: AgentEndEvent
    Result: List[AgentMessage] from end event
    """

    def __init__(self) -> None:
        super().__init__(
            is_complete=_is_agent_complete,
            extract_result=_extract_agent_result,
        )


def _is_agent_complete(event: AgentEvent) -> bool:
    """Check if event is terminal (AgentEndEvent)."""
    return event.type == "agent_end"


def _extract_agent_result(event: AgentEvent) -> List[AgentMessage]:
    """Extract final messages from terminal event."""
    if isinstance(event, AgentEndEvent):
        return event.messages
    raise ValueError(f"Unexpected event type for final result: {event.type}")


def create_agent_event_stream() -> AgentEventStream:
    """Factory function for AgentEventStream."""
    return AgentEventStream()


# ╭────────────────────────────────────────────────────────────╮
# │  Message Helpers                                             │
# ╰────────────────────────────────────────────────────────────╯


def _has_tool_calls(message: Optional[AssistantMessage]) -> bool:
    """Check if assistant message contains tool calls."""
    if not message or not message.content:
        return False
    return any(isinstance(c, ToolCall) for c in message.content)


def _get_tool_calls(message: AssistantMessage) -> List[ToolCall]:
    """Extract all tool calls from assistant message."""
    if not message.content:
        return []
    return [c for c in message.content if isinstance(c, ToolCall)]


def _find_tool_by_name(context: AgentContext, name: str) -> Optional[Any]:
    """Find tool by name in context."""
    for tool in context.tools:
        if tool.name == name:
            return tool
    return None


def _convert_messages_to_llm_format(messages: List[AgentMessage]) -> List[dict]:
    """Convert AgentMessage list to LLM API format.

    Basic implementation - can be customized via config.convert_to_llm.
    """
    result = []
    for msg in messages:
        if isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                result.append({"role": "user", "content": msg.content})
            else:
                # Convert content blocks
                content_list = []
                for c in msg.content:
                    if isinstance(c, TextContent):
                        content_list.append({"type": "text", "text": c.text})
                    elif hasattr(c, "data") and c.data:  # ImageContent
                        content_list.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{c.mimeType};base64,{c.data}"
                                },
                            }
                        )
                result.append({"role": "user", "content": content_list})

        elif isinstance(msg, AssistantMessage):
            content_str = ""
            tool_calls = []
            for c in msg.content:
                if isinstance(c, TextContent):
                    content_str += c.text
                elif isinstance(c, ToolCall):
                    tool_calls.append(
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": c.arguments},
                        }
                    )

            msg_dict: dict = {"role": "assistant", "content": content_str}
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls
            result.append(msg_dict)

        elif isinstance(msg, ToolResultMessage):
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.toolCallId,
                    "content": msg.content,
                }
            )

    return result


# ╭────────────────────────────────────────────────────────────╮
# │  Agent Loop Implementation                                   │
# ╰────────────────────────────────────────────────────────────╯


async def agent_loop(
    messages: List[AgentMessage],
    provider: Provider,
    model: str,
    context: Optional[AgentContext] = None,
    config: Optional[AgentLoopConfig] = None,
    signal: Optional[AbortSignal] = None,
) -> AgentEventStream:
    """Run agent conversation loop.

    Orchestrates multi-turn conversation with optional tool execution.
    Emits lifecycle events for monitoring and control.

    Args:
        messages: Initial messages to start conversation
        provider: LLM provider to use for generation
        model: Model identifier to use
        context: Optional agent context (system_prompt, tools, etc.)
        config: Optional loop configuration (limits, hooks, etc.)
        signal: Optional abort signal for cancellation

    Returns:
        AgentEventStream that can be:
        - Iterated: async for event in stream
        - Awaited: result = await stream.result() (returns final messages)

    Example:
        stream = await agent_loop(
            messages=[UserMessage(content="Calculate 2+2")],
            provider=provider,
            model="gpt-4",
            context=AgentContext(tools=[calculator_tool]),
        )

        async for event in stream:
            if event.type == "tool_execution_start":
                print(f"Running: {event.tool_call.name}")

        final_messages = await stream.result()
    """
    # Initialize
    stream = AgentEventStream()
    ctx = context or AgentContext()
    cfg = config or AgentLoopConfig()

    # Apply transform_context hook if provided
    if cfg.transform_context:
        ctx = cfg.transform_context(ctx)

    # Build message list
    conversation: List[AgentMessage] = list(messages)

    # Emit agent start event
    stream.push(AgentStartEvent(context=ctx))

    # Start agent loop task
    asyncio.create_task(
        _run_agent_loop(stream, conversation, ctx, cfg, provider, model, signal)
    )

    return stream


async def _run_agent_loop(
    stream: AgentEventStream,
    conversation: List[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    provider: Provider,
    model: str,
    signal: Optional[AbortSignal],
) -> None:
    """Internal: Run the agent loop logic."""
    try:
        turn = 0
        total_tool_calls = 0

        while turn < config.max_turns:
            # Check abort signal - but still emit end event
            if signal and signal.aborted:
                # Emit agent end event before breaking
                stream.push(AgentEndEvent(messages=conversation))
                return

            turn += 1

            # Emit turn start event
            stream.push(TurnStartEvent(turn=turn))

            # Prepare messages for LLM
            if config.convert_to_llm:
                llm_messages = config.convert_to_llm(conversation)
            else:
                llm_messages = _convert_messages_to_llm_format(conversation)

            # Add system prompt if present
            if context.system_prompt and turn == 1:
                llm_messages.insert(
                    0, {"role": "system", "content": context.system_prompt}
                )

            # Check abort before streaming
            if signal and signal.aborted:
                break

            # Stream from provider
            provider_stream = await provider.stream(llm_messages, model)

            # Collect assistant message from stream
            assistant_message: Optional[AssistantMessage] = None
            async for event in provider_stream:
                # Check abort during streaming
                if signal and signal.aborted:
                    break

                if is_terminal_event(event):
                    assistant_message = (
                        event.message if hasattr(event, "message") else None
                    )
                    break

            # Check if aborted during streaming
            if signal and signal.aborted:
                break

            if assistant_message is None:
                # No message received (error or aborted)
                break

            # Add assistant message to conversation
            conversation.append(assistant_message)

            # Check for tool calls
            tool_calls = _get_tool_calls(assistant_message)

            if tool_calls:
                # Check tool call limit
                total_tool_calls += len(tool_calls)
                if total_tool_calls > config.max_tool_calls:
                    # Too many tool calls, stop
                    break

                # Execute tools
                if config.tool_mode == "parallel":
                    await _execute_tools_parallel(
                        stream, tool_calls, context, config, conversation, signal
                    )
                else:
                    await _execute_tools_sequential(
                        stream, tool_calls, context, config, conversation, signal
                    )

                # Continue to next turn for tool results
                # Emit turn end
                stream.push(TurnEndEvent(turn=turn, message=assistant_message))
                continue

            # No tool calls - turn complete
            stream.push(TurnEndEvent(turn=turn, message=assistant_message))

            # Check stop reason
            if assistant_message.stopReason == StopReason.END_TURN:
                # Normal completion
                break
            elif assistant_message.stopReason == StopReason.MAX_TOKENS:
                # Max tokens reached, could continue but let's stop
                break
            # Otherwise continue to next turn

        # Emit agent end event
        stream.push(AgentEndEvent(messages=conversation))

    except Exception as e:
        # Error in agent loop
        stream.error(e)


async def _execute_tools_sequential(
    stream: AgentEventStream,
    tool_calls: List[ToolCall],
    context: AgentContext,
    config: AgentLoopConfig,
    conversation: List[AgentMessage],
    signal: Optional[AbortSignal],
) -> None:
    """Execute tools sequentially (one at a time)."""
    for tool_call in tool_calls:
        # Check abort
        if signal and signal.aborted:
            break

        # Emit start event
        stream.push(ToolExecutionStartEvent(tool_call=tool_call))

        # Run before_tool_call hook
        if config.before_tool_call:
            try:
                hook_result = await config.before_tool_call(tool_call)
                if hook_result is not None:
                    # Hook provided a result, skip actual execution
                    stream.push(
                        ToolExecutionEndEvent(
                            tool_call=tool_call,
                            result=hook_result,
                        )
                    )
                    # Add tool result message
                    tool_result_msg = ToolResultMessage(
                        toolCallId=tool_call.id,
                        content=str(hook_result),
                        isError=False,
                    )
                    conversation.append(tool_result_msg)
                    continue
            except Exception as e:
                # Hook error - treat as tool error
                stream.push(
                    ToolExecutionEndEvent(
                        tool_call=tool_call,
                        error=str(e),
                    )
                )
                tool_result_msg = ToolResultMessage(
                    toolCallId=tool_call.id,
                    content=f"Hook error: {str(e)}",
                    isError=True,
                )
                conversation.append(tool_result_msg)
                continue

        # Find tool
        tool = _find_tool_by_name(context, tool_call.name)

        if not tool:
            # Tool not found
            error_msg = f"Tool not found: {tool_call.name}"
            stream.push(
                ToolExecutionEndEvent(
                    tool_call=tool_call,
                    error=error_msg,
                )
            )
            tool_result_msg = ToolResultMessage(
                toolCallId=tool_call.id,
                content=error_msg,
                isError=True,
            )
            conversation.append(tool_result_msg)
            continue

        # Execute tool
        try:
            result = await tool.execute(tool_call.arguments)

            # Run after_tool_call hook
            if config.after_tool_call:
                try:
                    await config.after_tool_call(tool_call, result)
                except Exception:
                    # Ignore hook errors
                    pass

            # Emit success event
            stream.push(
                ToolExecutionEndEvent(
                    tool_call=tool_call,
                    result=result,
                )
            )

            # Add tool result message
            content = result if isinstance(result, str) else str(result)
            tool_result_msg = ToolResultMessage(
                toolCallId=tool_call.id,
                content=content,
                isError=False,
            )
            conversation.append(tool_result_msg)

        except Exception as e:
            # Tool execution error
            error_msg = str(e)
            stream.push(
                ToolExecutionEndEvent(
                    tool_call=tool_call,
                    error=error_msg,
                )
            )
            tool_result_msg = ToolResultMessage(
                toolCallId=tool_call.id,
                content=error_msg,
                isError=True,
            )
            conversation.append(tool_result_msg)


async def _execute_tools_parallel(
    stream: AgentEventStream,
    tool_calls: List[ToolCall],
    context: AgentContext,
    config: AgentLoopConfig,
    conversation: List[AgentMessage],
    signal: Optional[AbortSignal],
) -> None:
    """Execute tools in parallel (all at once)."""

    async def execute_single(
        tool_call: ToolCall,
    ) -> tuple[ToolCall, Any, Optional[str]]:
        """Execute single tool and return result."""
        # Check abort
        if signal and signal.aborted:
            return (tool_call, None, "Aborted")

        # Emit start event
        stream.push(ToolExecutionStartEvent(tool_call=tool_call))

        # Run before_tool_call hook
        if config.before_tool_call:
            try:
                hook_result = await config.before_tool_call(tool_call)
                if hook_result is not None:
                    # Hook provided result
                    stream.push(
                        ToolExecutionEndEvent(
                            tool_call=tool_call,
                            result=hook_result,
                        )
                    )
                    return (tool_call, hook_result, None)
            except Exception as e:
                # Hook error
                error_msg = f"Hook error: {str(e)}"
                stream.push(
                    ToolExecutionEndEvent(
                        tool_call=tool_call,
                        error=error_msg,
                    )
                )
                return (tool_call, None, error_msg)

        # Find tool
        tool = _find_tool_by_name(context, tool_call.name)

        if not tool:
            # Tool not found
            error_msg = f"Tool not found: {tool_call.name}"
            stream.push(
                ToolExecutionEndEvent(
                    tool_call=tool_call,
                    error=error_msg,
                )
            )
            return (tool_call, None, error_msg)

        # Execute tool
        try:
            result = await tool.execute(tool_call.arguments)

            # Run after_tool_call hook
            if config.after_tool_call:
                try:
                    await config.after_tool_call(tool_call, result)
                except Exception:
                    pass

            # Emit success event
            stream.push(
                ToolExecutionEndEvent(
                    tool_call=tool_call,
                    result=result,
                )
            )

            return (tool_call, result, None)

        except Exception as e:
            # Tool execution error
            error_msg = str(e)
            stream.push(
                ToolExecutionEndEvent(
                    tool_call=tool_call,
                    error=error_msg,
                )
            )
            return (tool_call, None, error_msg)

    # Execute all tools in parallel
    tasks = [execute_single(tc) for tc in tool_calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and build tool result messages
    for i, result in enumerate(results):
        tool_call = tool_calls[i]

        if isinstance(result, Exception):
            # Task failed
            error_msg = f"Execution failed: {str(result)}"
            tool_result_msg = ToolResultMessage(
                toolCallId=tool_call.id,
                content=error_msg,
                isError=True,
            )
        else:
            tc, res, err = result
            if err:
                # Error during execution
                tool_result_msg = ToolResultMessage(
                    toolCallId=tool_call.id,
                    content=err,
                    isError=True,
                )
            else:
                # Success
                content = res if isinstance(res, str) else str(res)
                tool_result_msg = ToolResultMessage(
                    toolCallId=tool_call.id,
                    content=content,
                    isError=False,
                )

        conversation.append(tool_result_msg)
