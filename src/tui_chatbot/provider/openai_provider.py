"""OpenAI Provider 实现."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import ConfigDict, Field

from ..agent.types import (
    AgentEvent,
    AgentEventType,
    AssistantMessage,
    ChatResult,
    TextContent,
    ToolCallContent,
    ToolCallMessage,
    UserMessage,
)
from ..core.event_stream import EventStream
from .base import Provider, ProviderConfig

if TYPE_CHECKING:
    from openai import AsyncOpenAI


ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class OpenAIProviderConfig(ProviderConfig):
    """OpenAI Provider 配置."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(default="gpt-3.5-turbo")
    reasoning_effort: Optional[ReasoningEffort] = Field(default=None)
    temperature: Optional[float] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    top_p: Optional[float] = Field(default=None)
    frequency_penalty: Optional[float] = Field(default=None)
    presence_penalty: Optional[float] = Field(default=None)
    stream: bool = Field(default=True)


class OpenAIProvider(Provider):
    """OpenAI API 提供商实现.

    支持流式输出、工具调用、推理内容 (reasoning_content)。

    Usage:
        from openai import AsyncOpenAI
        from tui_chatbot.provider import OpenAIProvider, OpenAIProviderConfig

        client = AsyncOpenAI(api_key="...")
        config = OpenAIProviderConfig(model="gpt-4")
        provider = OpenAIProvider(client, config)

        stream = await provider.stream_chat(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )

        async for event in stream:
            print(event)

        result = await stream.result()
    """

    def __init__(
        self,
        client: Optional["AsyncOpenAI"] = None,
        config: Optional[OpenAIProviderConfig] = None,
    ):
        """初始化 OpenAI Provider.

        Args:
            client: OpenAI 异步客户端，如果为 None 则从环境变量创建
            config: 提供商配置，如果为 None 则使用默认配置
        """
        self._config = config or OpenAIProviderConfig()

        if client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self._config.base_url,
                api_key=self._config.api_key or os.getenv("OPENAI_API_KEY", ""),
            )
        else:
            self._client = client

    @property
    def name(self) -> str:
        """提供商名称."""
        return "openai"

    @property
    def api_type(self) -> str:
        """API 类型标识."""
        return "openai-chat"

    @property
    def config(self) -> OpenAIProviderConfig:
        """获取当前配置."""
        return self._config

    def _build_messages(self, messages: List[dict]) -> List[dict]:
        """构建标准 OpenAI 消息格式."""
        return messages

    def _build_tools(self, tools: Optional[List[dict]]) -> Optional[List[dict]]:
        """构建 OpenAI 工具定义."""
        if not tools:
            return None
        return tools

    async def stream_chat(
        self,
        model: str,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        signal: Optional["AbortSignal"] = None,  # type: ignore  # noqa: F821
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> EventStream[AgentEvent, ChatResult]:
        """流式对话实现.

        返回 EventStream 推送 AgentEvent 事件，包括:
        - MESSAGE_START: 消息开始
        - MESSAGE_UPDATE: 内容流式更新 (包括 reasoning_content)
        - MESSAGE_END: 消息结束
        - TOOL_EXECUTION_START: 工具调用开始
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

                # 构建请求参数
                create_params: Dict[str, Any] = {
                    "model": model,
                    "messages": self._build_messages(messages),
                    "stream": True,
                }

                # 添加可选参数
                if tools:
                    create_params["tools"] = self._build_tools(tools)

                if self._config.reasoning_effort:
                    create_params["reasoning_effort"] = self._config.reasoning_effort

                # 使用传入的参数，如果为 None 则使用配置中的值
                effective_temperature = (
                    temperature if temperature is not None else self._config.temperature
                )
                if effective_temperature is not None:
                    create_params["temperature"] = effective_temperature

                effective_max_tokens = (
                    max_tokens if max_tokens is not None else self._config.max_tokens
                )
                if effective_max_tokens is not None:
                    create_params["max_tokens"] = effective_max_tokens

                # 开始流式请求
                api_stream = await self._client.chat.completions.create(**create_params)

                # 流式处理变量
                content_parts: List[str] = []
                reasoning_parts: List[str] = []
                tool_calls: List[Dict[str, Any]] = []
                message_started = False
                finish_reason: Optional[str] = None

                async for chunk in api_stream:
                    # 检查取消信号
                    if signal and signal.aborted:  # type: ignore  # noqa: F821
                        raise asyncio.CancelledError(f"Aborted: {signal.reason}")  # type: ignore  # noqa: F821

                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta
                    if not delta:
                        continue

                    # 第一次收到内容时发送 MESSAGE_START
                    if not message_started:
                        stream.push(AgentEvent(type=AgentEventType.MESSAGE_START))
                        message_started = True

                    # 处理 reasoning_content (o1, o3 等推理模型)
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                        # 发送更新事件，包含推理内容
                        stream.push(
                            AgentEvent(
                                type=AgentEventType.MESSAGE_UPDATE,
                                partial_result={
                                    "type": "reasoning",
                                    "content": reasoning,
                                },
                            )
                        )

                    # 处理 content
                    content = getattr(delta, "content", None)
                    if content:
                        content_parts.append(content)
                        stream.push(
                            AgentEvent(
                                type=AgentEventType.MESSAGE_UPDATE,
                                partial_result={
                                    "type": "content",
                                    "content": content,
                                },
                            )
                        )

                    # 处理工具调用
                    delta_tool_calls = getattr(delta, "tool_calls", None)
                    if delta_tool_calls:
                        for tc in delta_tool_calls:
                            tc_index = tc.index
                            # 确保 tool_calls 列表足够长
                            while len(tool_calls) <= tc_index:
                                tool_calls.append(
                                    {"id": "", "name": "", "arguments": ""}
                                )

                            if tc.id:
                                tool_calls[tc_index]["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    tool_calls[tc_index]["name"] = tc.function.name
                                if tc.function.arguments:
                                    tool_calls[tc_index]["arguments"] += (
                                        tc.function.arguments
                                    )

                    # 记录 finish_reason
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason

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
                                arguments=self._parse_tool_args(
                                    tc.get("arguments", "")
                                ),
                            )
                        )

                assistant_message = AssistantMessage(
                    content=message_content,
                    stop_reason=finish_reason,
                )

                # 发送 MESSAGE_END
                stream.push(
                    AgentEvent(
                        type=AgentEventType.MESSAGE_END,
                        message=assistant_message,
                    )
                )

                # 发送 TURN_END
                tool_call_messages = []
                for tc in tool_calls:
                    if tc.get("name"):
                        tool_call_messages.append(
                            ToolCallMessage(
                                id=tc.get("id", ""),
                                name=tc["name"],
                                arguments=self._parse_tool_args(
                                    tc.get("arguments", "")
                                ),
                            )
                        )

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
                    finish_reason=finish_reason,
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

    def _parse_tool_args(self, args_str: str) -> Dict[str, Any]:
        """解析工具参数 JSON."""
        if not args_str:
            return {}
        try:
            import json

            return json.loads(args_str)
        except json.JSONDecodeError:
            return {}

    async def list_models(self) -> List[str]:
        """列出可用模型."""
        try:
            models = await self._client.models.list()
            return [m.id for m in models.data]
        except Exception:
            return []

    async def simple_chat(
        self,
        messages: List[dict],
        model: Optional[str] = None,
    ) -> str:
        """简单非流式对话 (便捷方法).

        Args:
            messages: 消息列表
            model: 模型名称，默认使用配置的模型

        Returns:
            str: 助手回复内容
        """
        model = model or self._config.model

        create_params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        if self._config.temperature is not None:
            create_params["temperature"] = self._config.temperature

        if self._config.max_tokens is not None:
            create_params["max_tokens"] = self._config.max_tokens

        response = await self._client.chat.completions.create(**create_params)

        if response.choices and response.choices[0].message:
            return response.choices[0].message.content or ""
        return ""
