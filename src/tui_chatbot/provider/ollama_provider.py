"""Ollama Provider 实现 - 本地模型支持."""

from __future__ import annotations

import asyncio
import json
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
    import httpx


class OllamaProviderConfig(ProviderConfig):
    """Ollama Provider 配置."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(default="llama3.2")
    base_url: str = Field(default="http://localhost:11434")
    temperature: Optional[float] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    top_p: Optional[float] = Field(default=None)


class OllamaProvider(Provider):
    """Ollama 本地模型提供商实现.

    支持本地部署的 LLM 模型，通过 Ollama HTTP API 访问。

    Usage:
        from tui_chatbot.provider import OllamaProvider, OllamaProviderConfig

        config = OllamaProviderConfig(model="llama3.2", base_url="http://localhost:11434")
        provider = OllamaProvider(config)

        stream = await provider.stream_chat(
            model="llama3.2",
            messages=[{"role": "user", "content": "Hello"}]
        )

        async for event in stream:
            print(event)
    """

    def __init__(
        self,
        config: Optional[OllamaProviderConfig] = None,
        client: Optional["httpx.AsyncClient"] = None,
    ):
        self._config = config or OllamaProviderConfig()
        self._client = client

    @property
    def name(self) -> str:
        """提供商名称."""
        return "ollama"

    @property
    def api_type(self) -> str:
        """API 类型标识."""
        return "ollama"

    async def _get_client(self) -> "httpx.AsyncClient":
        """获取或创建 httpx 客户端."""
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self._config.base_url.rstrip("/"),
                timeout=300.0,
            )
        return self._client

    async def stream_chat(
        self,
        model: str,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        signal: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> EventStream[AgentEvent, ChatResult]:
        """流式对话实现.

        调用 Ollama Chat API (/api/chat)，支持流式响应。

        Args:
            model: 模型名称
            messages: 消息列表
            tools: 可选的工具定义列表
            signal: 可选的取消信号
            temperature: 温度参数，控制生成随机性
            max_tokens: 最大生成 token 数

        Returns:
            EventStream[AgentEvent, ChatResult]: 流式事件流
        """
        stream = EventStream[AgentEvent, ChatResult]()

        async def _stream():
            content_parts: List[str] = []

            try:
                # 检查取消信号
                if signal and signal.aborted:
                    stream.end(ChatResult(messages=[]))
                    return

                client = await self._get_client()

                # 构建 Ollama 请求体
                payload = {
                    "model": model or self._config.model,
                    "messages": messages,
                    "stream": True,
                }

                # 添加可选参数
                options = {}
                temp = (
                    temperature if temperature is not None else self._config.temperature
                )
                if temp is not None:
                    options["temperature"] = temp

                max_tok = (
                    max_tokens if max_tokens is not None else self._config.max_tokens
                )
                if max_tok is not None:
                    options["num_predict"] = max_tok

                if self._config.top_p is not None:
                    options["top_p"] = self._config.top_p

                if options:
                    payload["options"] = options

                # Ollama 的工具支持
                if tools:
                    payload["tools"] = tools

                # 发送流式请求
                async with client.stream(
                    "POST",
                    "/api/chat",
                    json=payload,
                ) as response:
                    stream.push(AgentEvent(type=AgentEventType.MESSAGE_START))

                    async for line in response.aiter_lines():
                        if signal and signal.aborted:
                            raise asyncio.CancelledError("Aborted")

                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # 处理消息增量
                        message = data.get("message", {})
                        content = message.get("content", "")

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

                        # 检查是否完成
                        if data.get("done", False):
                            break

                # 构建最终消息
                final_content = "".join(content_parts)
                assistant_message = AssistantMessage(
                    content=[TextContent(text=final_content)] if final_content else [],
                    stop_reason="stop",
                )

                stream.push(
                    AgentEvent(
                        type=AgentEventType.MESSAGE_END,
                        message=assistant_message,
                    )
                )

                stream.push(
                    AgentEvent(
                        type=AgentEventType.TURN_END,
                        message=assistant_message,
                        tool_results=[],
                    )
                )

                result = ChatResult(
                    messages=[assistant_message],
                    finish_reason="stop",
                )
                stream.end(result)

            except asyncio.CancelledError:
                final_content = "".join(content_parts) if content_parts else ""
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
        """列出本地可用的 Ollama 模型.

        调用 Ollama API /api/tags 获取可用模型列表。

        Returns:
            List[str]: 模型名称列表
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    async def close(self) -> None:
        """关闭 HTTP 客户端."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
