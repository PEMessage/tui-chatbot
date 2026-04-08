"""Anthropic (Claude) Provider 实现."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from ..agent.types import (
    AgentEvent,
    AgentEventType,
    AssistantMessage,
    ChatResult,
    TextContent,
    ToolCallContent,
)
from ..core.event_stream import EventStream
from .base import Provider, ProviderConfig

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic


class AnthropicProviderConfig(ProviderConfig):
    """Anthropic Provider 配置."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(default="claude-3-sonnet-20240229")
    max_tokens: Optional[int] = Field(default=4096)
    temperature: Optional[float] = Field(default=None)
    top_p: Optional[float] = Field(default=None)


class AnthropicProvider(Provider):
    """Anthropic Claude API 提供商实现.

    支持流式输出、工具调用。

    Usage:
        from anthropic import AsyncAnthropic
        from tui_chatbot.provider import AnthropicProvider, AnthropicProviderConfig

        client = AsyncAnthropic(api_key="...")
        config = AnthropicProviderConfig(model="claude-3-opus-20240229")
        provider = AnthropicProvider(client, config)

        stream = await provider.stream_chat(
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Hello"}]
        )

        async for event in stream:
            print(event)

        result = await stream.result()
    """

    def __init__(
        self,
        client: Optional["AsyncAnthropic"] = None,
        config: Optional[AnthropicProviderConfig] = None,
    ):
        """初始化 Anthropic Provider.

        Args:
            client: Anthropic 异步客户端，如果为 None 则从环境变量创建
            config: 提供商配置，如果为 None 则使用默认配置
        """
        self._config = config or AnthropicProviderConfig()

        if client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(
                base_url=self._config.base_url or None,
                api_key=self._config.api_key or os.getenv("ANTHROPIC_API_KEY", ""),
            )
        else:
            self._client = client

    @property
    def name(self) -> str:
        """提供商名称."""
        return "Anthropic"

    @property
    def api_type(self) -> str:
        """API 类型标识."""
        return "anthropic"

    @property
    def config(self) -> AnthropicProviderConfig:
        """获取当前配置."""
        return self._config

    def _convert_messages(
        self, messages: List[dict]
    ) -> tuple[Optional[str], List[dict]]:
        """将标准消息格式转换为 Anthropic 格式.

        Anthropic 格式:
        - system: 独立的 system 参数 (字符串)
        - messages: 只有 user 和 assistant 角色的消息列表

        Args:
            messages: 标准格式的消息列表

        Returns:
            (system_prompt, anthropic_messages): 系统提示和转换后的消息
        """
        system_prompt: Optional[str] = None
        anthropic_messages: List[dict] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Anthropic 使用独立的 system 参数
            if role == "system":
                if system_prompt is None:
                    system_prompt = content
                else:
                    system_prompt += f"\n\n{content}"
                continue

            # Anthropic 只支持 user 和 assistant 角色
            if role in ("user", "assistant"):
                anthropic_messages.append({"role": role, "content": content})
            elif role == "tool":
                # 工具结果转换为 user 消息
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": f"<tool_result>{content}</tool_result>",
                    }
                )

        return system_prompt, anthropic_messages

    def _convert_tools(self, tools: Optional[List[dict]]) -> Optional[List[dict]]:
        """将 OpenAI 格式工具定义转换为 Anthropic 格式.

        Anthropic 工具格式:
        {
            "name": "tool_name",
            "description": "...",
            "input_schema": {...}  # 对应 OpenAI 的 parameters
        }
        """
        if not tools:
            return None

        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                anthropic_tools.append(
                    {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {}),
                    }
                )

        return anthropic_tools if anthropic_tools else None

    async def stream_chat(
        self,
        model: str,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        signal: Optional["AbortSignal"] = None,  # type: ignore  # noqa: F821
    ) -> EventStream[AgentEvent, ChatResult]:
        """流式对话实现.

        返回 EventStream 推送 AgentEvent 事件，包括:
        - MESSAGE_START: 消息开始
        - MESSAGE_UPDATE: 内容流式更新
        - MESSAGE_END: 消息结束
        - TURN_END: 一轮结束

        Args:
            model: 模型名称
            messages: 消息列表
            tools: 可选的工具定义列表
            signal: 可选的取消信号

        Returns:
            EventStream[AgentEvent, ChatResult]: 流式事件流
        """
        stream = EventStream[AgentEvent, ChatResult]()

        async def _stream():
            try:
                # 检查取消信号
                if signal and signal.aborted:  # type: ignore  # noqa: F821
                    stream.push(
                        AgentEvent(
                            type=AgentEventType.ERROR,
                            error=f"Aborted: {signal.reason}",  # type: ignore  # noqa: F821
                        )
                    )
                    stream.end(ChatResult(messages=[]))
                    return

                # 转换消息格式
                system_prompt, anthropic_messages = self._convert_messages(messages)

                # 构建请求参数
                create_params: Dict[str, Any] = {
                    "model": model,
                    "messages": anthropic_messages,
                    "max_tokens": self._config.max_tokens or 4096,
                    "stream": True,
                }

                # 添加可选参数
                if system_prompt:
                    create_params["system"] = system_prompt

                anthropic_tools = self._convert_tools(tools)
                if anthropic_tools:
                    create_params["tools"] = anthropic_tools

                if self._config.temperature is not None:
                    create_params["temperature"] = self._config.temperature

                if self._config.top_p is not None:
                    create_params["top_p"] = self._config.top_p

                # 开始流式请求
                api_stream = await self._client.messages.create(**create_params)

                # 流式处理变量
                content_parts: List[str] = []
                tool_calls: List[Dict[str, Any]] = []
                message_started = False
                finish_reason: Optional[str] = None
                current_tool_use: Optional[Dict[str, Any]] = None

                async for event in api_stream:
                    # 检查取消信号
                    if signal and signal.aborted:  # type: ignore  # noqa: F821
                        raise asyncio.CancelledError(f"Aborted: {signal.reason}")  # type: ignore  # noqa: F821

                    event_type = event.type

                    # 第一次收到内容时发送 MESSAGE_START
                    if not message_started and event_type in (
                        "content_block_start",
                        "content_block_delta",
                    ):
                        stream.push(AgentEvent(type=AgentEventType.MESSAGE_START))
                        message_started = True

                    # 处理内容块开始 (可能是文本或工具调用)
                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block:
                            block_type = getattr(block, "type", None)
                            if block_type == "tool_use":
                                # 新工具调用开始
                                current_tool_use = {
                                    "id": getattr(block, "id", ""),
                                    "name": getattr(block, "name", ""),
                                    "input": "",
                                }

                    # 处理内容增量
                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            # 处理文本增量
                            text = getattr(delta, "text", None)
                            if text:
                                content_parts.append(text)
                                stream.push(
                                    AgentEvent(
                                        type=AgentEventType.MESSAGE_UPDATE,
                                        partial_result={
                                            "type": "content",
                                            "content": text,
                                        },
                                    )
                                )

                            # 处理工具输入增量
                            partial_json = getattr(delta, "partial_json", None)
                            if partial_json and current_tool_use:
                                current_tool_use["input"] += partial_json

                    # 处理内容块结束
                    elif event_type == "content_block_stop":
                        if current_tool_use:
                            # 工具调用完成，解析参数
                            import json

                            input_str = current_tool_use.get("input", "")
                            try:
                                arguments = json.loads(input_str) if input_str else {}
                            except json.JSONDecodeError:
                                arguments = {}

                            tool_calls.append(
                                {
                                    "id": current_tool_use.get("id", ""),
                                    "name": current_tool_use.get("name", ""),
                                    "arguments": arguments,
                                }
                            )
                            current_tool_use = None

                    # 记录停止原因
                    elif event_type == "message_stop":
                        message = getattr(event, "message", None)
                        if message:
                            stop_reason = getattr(message, "stop_reason", None)
                            if stop_reason:
                                finish_reason = str(stop_reason)

                # 构建最终消息
                message_content: List[Any] = []

                # 添加文本内容
                final_content = "".join(content_parts)
                if final_content:
                    message_content.append(TextContent(text=final_content))

                # 添加工具调用
                for tc in tool_calls:
                    if tc.get("name"):
                        message_content.append(
                            ToolCallContent(
                                id=tc.get("id", ""),
                                name=tc["name"],
                                arguments=tc.get("arguments", {}),
                            )
                        )

                assistant_message = AssistantMessage(
                    content=message_content,
                    stop_reason=finish_reason or "end_turn",
                )

                # 发送 MESSAGE_END
                stream.push(
                    AgentEvent(
                        type=AgentEventType.MESSAGE_END,
                        message=assistant_message,
                    )
                )

                # 发送 TURN_END
                stream.push(
                    AgentEvent(
                        type=AgentEventType.TURN_END,
                        message=assistant_message,
                        tool_results=[],  # 将在 agent_loop 中填充
                    )
                )

                # 完成
                result = ChatResult(
                    messages=[assistant_message],
                    finish_reason=finish_reason or "end_turn",
                )
                stream.end(result)

            except asyncio.CancelledError:
                # 处理取消 - 返回已收集的内容
                final_content = (
                    "".join(content_parts) if "content_parts" in dir() else ""
                )
                if final_content:
                    assistant_message = AssistantMessage(
                        content=[TextContent(text=final_content)],
                        stop_reason="cancelled",
                    )
                    stream.push(
                        AgentEvent(
                            type=AgentEventType.MESSAGE_END,
                            message=assistant_message,
                        )
                    )
                    stream.end(
                        ChatResult(
                            messages=[assistant_message], finish_reason="cancelled"
                        )
                    )
                else:
                    stream.end(ChatResult(messages=[], finish_reason="cancelled"))
                raise

            except Exception as e:
                stream.push(
                    AgentEvent(
                        type=AgentEventType.ERROR,
                        error=str(e),
                    )
                )
                stream.error(e)

        # 启动后台任务
        asyncio.create_task(_stream())
        return stream

    async def list_models(self) -> List[str]:
        """列出可用模型.

        Returns:
            List[str]: 支持的 Claude 模型 ID 列表
        """
        # Anthropic 目前没有模型列表 API，返回已知支持的模型
        return [
            "claude-3-opus-20240229",
            "claude-3-opus-latest",
            "claude-3-sonnet-20240229",
            "claude-3-sonnet-latest",
            "claude-3-haiku-20240307",
            "claude-3-haiku-latest",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-20241022",
            "claude-3-5-haiku-latest",
            "claude-3-7-sonnet-20250219",
            "claude-3-7-sonnet-latest",
        ]
