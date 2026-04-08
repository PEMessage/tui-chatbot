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
    TextContent,
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
                            tool_results=[
                                _tool_result_to_dict(tr) for tr in tool_results
                            ],
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
    流式获取助手响应，集成真实 Provider 调用。

    流程:
    1. 从 config 获取 Provider (或从 ProviderRegistry)
    2. 将 context 转换为 Provider 需要的格式
    3. 调用 provider.stream_chat()
    4. 转发所有事件到 emit
    5. 构造并返回 AssistantMessage
    """
    # ═══════════════════════════════════════════════════════════════
    # 1. 获取 Provider
    # ═══════════════════════════════════════════════════════════════
    provider = _get_provider(config)

    # ═══════════════════════════════════════════════════════════════
    # 2. 转换消息格式
    # ═══════════════════════════════════════════════════════════════
    provider_messages = _convert_messages_to_provider_format(
        context, config.system_prompt
    )

    # ═══════════════════════════════════════════════════════════════
    # 3. 准备工具定义
    # ═══════════════════════════════════════════════════════════════
    tools_schema = None
    if config.tool_registry:
        tools_schema = config.tool_registry.to_openai_tools()

    # ═══════════════════════════════════════════════════════════════
    # 4. 调用 Provider 获取流式响应
    # ═══════════════════════════════════════════════════════════════
    stream = await provider.stream_chat(
        model=config.model,
        messages=provider_messages,
        tools=tools_schema,
        signal=signal,
    )

    # ═══════════════════════════════════════════════════════════════
    # 5. 处理流式事件，构建 AssistantMessage
    # ═══════════════════════════════════════════════════════════════
    content_parts: List[Any] = []
    current_text = ""
    current_tool_calls: Dict[str, Dict[str, Any]] = {}
    finish_reason: Optional[str] = None
    message_started = False

    async for event in stream:
        # 转发事件到 emit
        await emit(event)

        # 收集内容构建 AssistantMessage
        if event.type == AgentEventType.MESSAGE_START:
            message_started = True

        elif event.type == AgentEventType.MESSAGE_UPDATE:
            partial = event.partial_result or {}
            update_type = partial.get("type")

            if update_type == "content":
                content_chunk = partial.get("content", "")
                current_text += content_chunk

            elif update_type == "reasoning":
                # 推理内容暂不收集到 AssistantMessage，但已转发事件
                pass

        elif event.type == AgentEventType.MESSAGE_END:
            # 从事件中获取完整消息（如果有）
            if event.message and isinstance(event.message, AssistantMessage):
                # 使用事件中的完整消息
                content_parts = list(event.message.content)
                finish_reason = event.message.stop_reason

        elif event.type == AgentEventType.TURN_END:
            # TURN_END 也包含消息信息
            if event.message and isinstance(event.message, AssistantMessage):
                content_parts = list(event.message.content)
                finish_reason = event.message.stop_reason

    # 如果没有从事件获取到完整消息，自己构建
    if not content_parts:
        # 添加文本内容
        if current_text:
            content_parts.append(TextContent(text=current_text))

        # 添加工具调用（从流结果中获取）
        try:
            result = await stream.result()
            if result and result.messages:
                for msg in result.messages:
                    if isinstance(msg, AssistantMessage):
                        for content in msg.content:
                            if isinstance(content, ToolCallContent):
                                content_parts.append(content)
                        if msg.stop_reason:
                            finish_reason = msg.stop_reason
        except Exception:
            # 流可能已经结束或出错，忽略
            pass

    # 构建最终的 AssistantMessage
    assistant_msg = AssistantMessage(
        content=content_parts,
        stop_reason=finish_reason or "stop",
    )

    return assistant_msg


def _get_provider(config: AgentLoopConfig) -> "Provider":
    """从配置或注册表获取 Provider."""
    # 优先使用 config 中指定的 provider
    if config.provider:
        return config.provider

    # 回退到 ProviderRegistry
    from ..provider.registry import ProviderRegistry

    provider = ProviderRegistry.get("openai")
    if not provider:
        raise RuntimeError(
            "No provider available. Please set config.provider or register a provider in ProviderRegistry."
        )

    return provider


def _convert_messages_to_provider_format(
    messages: List[AgentMessage], system_prompt: str
) -> List[dict]:
    """将 AgentMessage 转换为 Provider 接受的字典格式."""
    result: List[dict] = []

    # 添加系统提示（如果有）
    if system_prompt:
        result.append({"role": "system", "content": system_prompt})

    for msg in messages:
        if isinstance(msg, UserMessage):
            result.append({"role": "user", "content": msg.content})

        elif isinstance(msg, AssistantMessage):
            # 处理助手消息（包含文本和工具调用）
            content_parts = []
            tool_calls = []

            for content in msg.content:
                if isinstance(content, TextContent):
                    content_parts.append({"type": "text", "text": content.text})
                elif isinstance(content, ToolCallContent):
                    tool_calls.append(
                        {
                            "id": content.id,
                            "type": "function",
                            "function": {
                                "name": content.name,
                                "arguments": content.arguments,
                            },
                        }
                    )

            if tool_calls:
                # 有工具调用时，content 为空或 null
                result.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    }
                )
            elif content_parts:
                # 纯文本消息
                text_content = "".join([p.get("text", "") for p in content_parts])
                result.append({"role": "assistant", "content": text_content})

        elif isinstance(msg, ToolResultMessage):
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                }
            )

    return result


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


def _tool_result_to_dict(result: ToolResultMessage) -> Dict[str, Any]:
    """将 ToolResultMessage 转换为字典格式."""
    return {
        "role": result.role,
        "tool_call_id": result.tool_call_id,
        "tool_name": result.tool_name,
        "content": result.content,
        "is_error": result.is_error,
        "details": result.details,
    }
