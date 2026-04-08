"""Agent Loop - 双层循环架构."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Awaitable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.event_stream import EventStream

from .types import (
    AgentEvent,
    AgentEventType,
    AgentLoopConfig,
    AgentMessage,
    AssistantMessage,
    ToolCallContent,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
    ChatResult,
    ToolExecutionMode,
)
from .tool import ToolRegistry, ToolResult


# ═══════════════════════════════════════════════════════════════
# Type Aliases
# ═══════════════════════════════════════════════════════════════

AgentEventSink = Callable[[AgentEvent], Awaitable[None]]
AbortSignal = Any  # 简化版 - 实际项目可能使用 AbortController


# ═══════════════════════════════════════════════════════════════
# Agent Loop
# ═══════════════════════════════════════════════════════════════


async def agent_loop(
    prompt: AgentMessage,
    history: List[AgentMessage],
    config: AgentLoopConfig,
    signal: Optional[AbortSignal] = None,
    emit: Optional[AgentEventSink] = None,
) -> "EventStream[AgentEvent, List[AgentMessage]]":
    """
    启动 Agent 循环.

    双层循环结构：
    - 外层：处理 follow-up 消息 (迭代处理多轮对话)
    - 内层：处理单轮对话 (助手响应 + 工具调用)

    Args:
        prompt: 当前用户提示消息
        history: 历史消息列表
        config: Agent 循环配置
        signal: 取消信号
        emit: 事件回调函数

    Returns:
        EventStream: 事件流，包含 AgentEvent 和最终结果 List[AgentMessage]
    """
    # 延迟导入避免循环依赖
    from ..core.event_stream import EventStream

    stream = EventStream[AgentEvent, List[AgentMessage]]()

    async def _run():
        try:
            all_new_messages: List[AgentMessage] = [prompt]
            context = history + [prompt]

            # 发送 Agent 开始事件
            await _emit(AgentEvent(type=AgentEventType.AGENT_START))

            # 发送消息开始/结束事件 (针对 prompt)
            await _emit(AgentEvent(type=AgentEventType.MESSAGE_START, message=prompt))
            await _emit(AgentEvent(type=AgentEventType.MESSAGE_END, message=prompt))

            iteration = 0
            while iteration < config.max_iterations:
                iteration += 1

                # 内层循环：处理工具调用
                has_more_tool_calls = True
                turn_count = 0

                while has_more_tool_calls:
                    turn_count += 1
                    await _emit(AgentEvent(type=AgentEventType.TURN_START))

                    # 流式助手响应
                    assistant_msg = await _stream_assistant_response(
                        context, config, signal, _emit
                    )
                    all_new_messages.append(assistant_msg)

                    # 检查工具调用
                    tool_calls = _extract_tool_calls(assistant_msg)
                    has_more_tool_calls = len(tool_calls) > 0

                    tool_results: List[ToolResultMessage] = []
                    if has_more_tool_calls:
                        # 执行工具调用 (三阶段模式)
                        tool_results = await _execute_tool_calls(
                            tool_calls,
                            config.tool_registry,
                            signal,
                            _emit,
                            mode=config.tool_execution_mode,
                        )
                        context.extend(tool_results)
                        all_new_messages.extend(tool_results)

                    await _emit(
                        AgentEvent(
                            type=AgentEventType.TURN_END,
                            message=assistant_msg,
                            tool_results=tool_results,
                        )
                    )

                    # 检查是否需要继续 (简化版：暂时不支持 follow-up)
                    if not has_more_tool_calls:
                        break

                # 外层循环检查：简化版，第一轮后退出
                break

            # 发送 Agent 结束事件
            await _emit(
                AgentEvent(type=AgentEventType.AGENT_END, messages=all_new_messages)
            )
            stream.end(all_new_messages)

        except Exception as e:
            await _emit(AgentEvent(type=AgentEventType.ERROR, error=str(e)))
            stream.error(e)

    async def _emit(event: AgentEvent):
        """发送事件到流和回调."""
        stream.push(event)
        if emit:
            await emit(event)

    # 启动后台任务
    asyncio.create_task(_run())
    return stream


# ═══════════════════════════════════════════════════════════════
# Assistant Response Streaming
# ═══════════════════════════════════════════════════════════════


async def _stream_assistant_response(
    context: List[AgentMessage],
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
) -> AssistantMessage:
    """
    流式获取助手响应.

    这是一个简化版实现。实际项目中应该：
    1. 调用 Provider 进行 LLM 流式请求
    2. 处理流式返回的 tokens
    3. 构造 AssistantMessage
    """
    # 简化版：模拟一个助手响应
    # 实际实现应该通过 Provider 调用 LLM

    assistant_msg = AssistantMessage(content=[], stop_reason="stop")

    # 发送消息开始事件
    await emit(AgentEvent(type=AgentEventType.MESSAGE_START, message=assistant_msg))

    # 这里应该进行实际的 LLM 流式调用
    # 并发送 MESSAGE_UPDATE 事件

    # 发送消息结束事件
    await emit(AgentEvent(type=AgentEventType.MESSAGE_END, message=assistant_msg))

    return assistant_msg


# ═══════════════════════════════════════════════════════════════
# Tool Call Extraction
# ═══════════════════════════════════════════════════════════════


def _extract_tool_calls(message: AssistantMessage) -> List[ToolCallMessage]:
    """从助手消息中提取工具调用."""
    tool_calls: List[ToolCallMessage] = []

    for content in message.content:
        if isinstance(content, ToolCallContent):
            tool_calls.append(
                ToolCallMessage(
                    id=content.id, name=content.name, arguments=content.arguments
                )
            )

    return tool_calls


# ═══════════════════════════════════════════════════════════════
# Tool Execution - Three Phase Pattern
# ═══════════════════════════════════════════════════════════════


async def _execute_tool_calls(
    tool_calls: List[ToolCallMessage],
    registry: ToolRegistry,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
    mode: ToolExecutionMode = ToolExecutionMode.PARALLEL,
) -> List[ToolResultMessage]:
    """
    执行工具调用 - 三阶段模式.

    Phase 1: Sequential preflight (验证)
    Phase 2: Concurrent execution (并行执行)
    Phase 3: Sequential finalization (按原始顺序发送结果)

    Args:
        tool_calls: 工具调用列表
        registry: 工具注册表
        signal: 取消信号
        emit: 事件回调
        mode: 执行模式 (SEQUENTIAL 或 PARALLEL)

    Returns:
        工具结果消息列表 (按原始顺序)
    """
    if mode == ToolExecutionMode.SEQUENTIAL:
        return await _execute_sequential(tool_calls, registry, signal, emit)
    else:
        return await _execute_parallel(tool_calls, registry, signal, emit)


async def _execute_sequential(
    tool_calls: List[ToolCallMessage],
    registry: ToolRegistry,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
) -> List[ToolResultMessage]:
    """串行执行工具调用."""
    results: List[ToolResultMessage] = []

    for tc in tool_calls:
        # Phase 1: Preflight (验证)
        tool = registry.get(tc.name)
        if not tool:
            result_msg = ToolResultMessage(
                role="tool_result",
                tool_call_id=tc.id,
                content=f"Tool {tc.name} not found",
                is_error=True,
            )
            await _emit_tool_execution_start(emit, tc)
            await _emit_tool_execution_end(emit, tc, result_msg, True)
            results.append(result_msg)
            continue

        try:
            validated = tool.validate_params(tc.arguments)
        except Exception as e:
            result_msg = ToolResultMessage(
                role="tool_result",
                tool_call_id=tc.id,
                content=f"Parameter validation failed: {e}",
                is_error=True,
            )
            await _emit_tool_execution_start(emit, tc)
            await _emit_tool_execution_end(emit, tc, result_msg, True)
            results.append(result_msg)
            continue

        # Phase 2: Execution
        await _emit_tool_execution_start(emit, tc)

        try:
            result = await tool.execute(validated, signal)
            is_error = result.is_error
        except Exception as e:
            result = ToolResult(content=str(e), is_error=True)
            is_error = True

        # Phase 3: Finalization
        result_msg = ToolResultMessage(
            role="tool_result",
            tool_call_id=tc.id,
            tool_name=tc.name,
            content=result.content,
            is_error=is_error,
            details=result.details,
        )

        await _emit_tool_execution_end(emit, tc, result_msg, is_error)

        # 发送消息事件
        await emit(AgentEvent(type=AgentEventType.MESSAGE_START, message=result_msg))
        await emit(AgentEvent(type=AgentEventType.MESSAGE_END, message=result_msg))

        results.append(result_msg)

    return results


async def _execute_parallel(
    tool_calls: List[ToolCallMessage],
    registry: ToolRegistry,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
) -> List[ToolResultMessage]:
    """并行执行工具调用."""
    results: List[ToolResultMessage] = []

    # Phase 1: Sequential preflight (验证)
    prepared: List[tuple] = []  # (tool_call, tool, validated_params)

    for tc in tool_calls:
        await _emit_tool_execution_start(emit, tc)

        tool = registry.get(tc.name)
        if not tool:
            result_msg = ToolResultMessage(
                role="tool_result",
                tool_call_id=tc.id,
                tool_name=tc.name,
                content=f"Tool {tc.name} not found",
                is_error=True,
            )
            await _emit_tool_execution_end(emit, tc, result_msg, True)
            results.append((tc.id, result_msg))
            continue

        try:
            validated = tool.validate_params(tc.arguments)
            prepared.append((tc, tool, validated))
        except Exception as e:
            result_msg = ToolResultMessage(
                role="tool_result",
                tool_call_id=tc.id,
                tool_name=tc.name,
                content=f"Parameter validation failed: {e}",
                is_error=True,
            )
            await _emit_tool_execution_end(emit, tc, result_msg, True)
            results.append((tc.id, result_msg))

    # Phase 2: Concurrent execution
    async def execute_one(tc: ToolCallMessage, tool: Any, validated: Any) -> tuple:
        """执行单个工具调用."""
        try:
            result = await tool.execute(validated, signal)
            return (tc, result, False)
        except Exception as e:
            return (tc, ToolResult(content=str(e), is_error=True), True)

    if prepared:
        executed = await asyncio.gather(
            *[execute_one(tc, tool, validated) for tc, tool, validated in prepared]
        )
    else:
        executed = []

    # Phase 3: Sequential finalization (按原始顺序)
    # 构建结果映射以便按原始顺序输出
    execution_results: Dict[str, tuple] = {}
    for tc, result, is_error in executed:
        execution_results[tc.id] = (tc, result, is_error)

    # 按原始 tool_calls 顺序处理
    for tc in tool_calls:
        if tc.id in execution_results:
            original_tc, result, is_error = execution_results[tc.id]

            result_msg = ToolResultMessage(
                role="tool_result",
                tool_call_id=tc.id,
                tool_name=tc.name,
                content=result.content,
                is_error=is_error,
                details=result.details,
            )

            await _emit_tool_execution_end(emit, tc, result_msg, is_error)

            # 发送消息事件
            await emit(
                AgentEvent(type=AgentEventType.MESSAGE_START, message=result_msg)
            )
            await emit(AgentEvent(type=AgentEventType.MESSAGE_END, message=result_msg))

            results.append((tc.id, result_msg))

    # 返回按原始顺序排列的结果列表
    final_results: List[ToolResultMessage] = []
    result_map = {tid: msg for tid, msg in results}
    for tc in tool_calls:
        if tc.id in result_map:
            final_results.append(result_map[tc.id])

    return final_results


# ═══════════════════════════════════════════════════════════════
# Event Helpers
# ═══════════════════════════════════════════════════════════════


async def _emit_tool_execution_start(
    emit: AgentEventSink, tool_call: ToolCallMessage
) -> None:
    """发送工具执行开始事件."""
    await emit(
        AgentEvent(
            type=AgentEventType.TOOL_EXECUTION_START,
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            args=tool_call.arguments,
        )
    )


async def _emit_tool_execution_end(
    emit: AgentEventSink,
    tool_call: ToolCallMessage,
    result: ToolResultMessage,
    is_error: bool,
) -> None:
    """发送工具执行结束事件."""
    await emit(
        AgentEvent(
            type=AgentEventType.TOOL_EXECUTION_END,
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=result.content,
            is_error=is_error,
        )
    )
