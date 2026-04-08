"""重构后的 Daemon - 使用新架构的精简版本.

设计:
    - 使用 ProviderRegistry 获取 Provider
    - 使用 AgentEvent 事件系统
    - 保留 trim_history 逻辑
    - 支持工具注册和调用
    - 返回 EventStream 而非 AsyncGenerator
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, List, Dict, Optional, Any

from dotenv import load_dotenv

from .core.event_stream import EventStream
from .core.events import AgentEvent, AgentEventType, ChatResult, TokenStats
from .provider.registry import ProviderRegistry

if TYPE_CHECKING:
    from .agent.tool import ToolRegistry
    from .core.abort_controller import AbortSignal

load_dotenv()


SYSTEM_MSG = {"role": "system", "content": "You are a helpful assistant."}


class Daemon:
    """重构后的 Daemon - 精简版本.

    职责:
        - 管理对话历史
        - 通过 ProviderRegistry 调用 LLM
        - 支持工具调用
        - 返回 EventStream 事件流
    """

    def __init__(
        self,
        provider_api: str = "openai",
        model: Optional[str] = None,
        history_limit: int = 10,
        tool_registry: Optional["ToolRegistry"] = None,
    ):
        """初始化 Daemon.

        Args:
            provider_api: Provider API 类型标识 (如 'openai', 'openai-chat')
            model: 模型名称，如果为 None 使用 Provider 默认模型
            history_limit: 历史消息保留数量
            tool_registry: 可选的工具注册表
        """
        self._provider_api = provider_api
        self._model = model
        self._history_limit = history_limit
        self._tool_registry = tool_registry

        # 对话历史
        self._messages: List[Dict[str, str]] = [SYSTEM_MSG.copy()]

        # 初始化 Provider
        self._provider = ProviderRegistry.get(provider_api)

    @property
    def model(self) -> str:
        """当前模型名称."""
        return self._model or "unknown"

    @property
    def messages(self) -> List[Dict[str, str]]:
        """当前对话历史 (只读)."""
        return self._messages.copy()

    def _get_provider(self):
        """获取当前 Provider."""
        if self._provider is None:
            # 尝试重新获取 (可能在初始化后注册)
            self._provider = ProviderRegistry.get(self._provider_api)
        return self._provider

    def trim_history(self) -> None:
        """修剪历史消息，保留 system 消息和最近的 history_limit 条."""
        max_len = self._history_limit + 1  # +1 for system message
        if len(self._messages) > max_len:
            self._messages = [self._messages[0]] + self._messages[
                -self._history_limit :
            ]

    def switch_model(self, name: str) -> None:
        """切换模型."""
        self._model = name

    def clear_history(self) -> None:
        """清除对话历史，只保留 system 消息."""
        self._messages = [SYSTEM_MSG.copy()]

    def chat(
        self, text: str, signal: Optional["AbortSignal"] = None
    ) -> EventStream[AgentEvent, ChatResult]:
        """发送消息并获取流式响应.

        Args:
            text: 用户输入文本
            signal: 可选的取消信号

        Returns:
            EventStream: 事件流，包含 AgentEvent 和 ChatResult
        """
        stream = EventStream[AgentEvent, ChatResult]()

        async def _stream():
            try:
                # 检查 Provider
                provider = self._get_provider()
                if provider is None:
                    stream.push(
                        AgentEvent(
                            type=AgentEventType.ERROR,
                            error=f"Provider not found: {self._provider_api}",
                        )
                    )
                    stream.end(ChatResult(content="", messages=[]))
                    return

                # 检查取消信号
                if signal and signal.aborted:
                    stream.push(
                        AgentEvent(
                            type=AgentEventType.ERROR,
                            error=f"Aborted: {signal.reason}",
                        )
                    )
                    stream.end(ChatResult(content="", messages=[]))
                    return

                # 修剪历史并添加用户消息
                self.trim_history()
                self._messages.append({"role": "user", "content": text})

                # 准备工具
                tools = None
                if self._tool_registry:
                    tools = self._tool_registry.to_openai_tools()

                # 获取模型
                model = self._model or getattr(
                    provider, "default_model", "gpt-3.5-turbo"
                )

                # 发送 Agent 开始事件
                stream.push(AgentEvent(type=AgentEventType.AGENT_START))
                stream.push(AgentEvent(type=AgentEventType.TURN_START))

                # 调用 Provider 进行流式对话
                provider_stream = await provider.stream_chat(
                    model=model,
                    messages=self._messages,
                    tools=tools,
                    signal=signal,
                )

                # 统计信息
                import time

                stats = TokenStats()
                stats.start_time = time.time()

                # 转发 Provider 事件并收集响应
                content_parts = []
                reasoning_parts = []
                final_result = None

                async for event in provider_stream:
                    # 转发事件
                    stream.push(event)

                    # 收集内容
                    if event.type == AgentEventType.MESSAGE_UPDATE:
                        partial = getattr(event, "partial_result", None)
                        if partial:
                            if partial.get("type") == "reasoning":
                                reasoning_parts.append(partial.get("content", ""))
                                stats.on_token()
                                stats.r_tokens += len(partial.get("content", "")) // 4
                            elif partial.get("type") == "content":
                                content_parts.append(partial.get("content", ""))
                                stats.on_token()
                                stats.c_tokens += len(partial.get("content", "")) // 4

                    # 收集最终结果
                    if event.type == AgentEventType.TURN_END:
                        if hasattr(event, "message") and event.message:
                            # 从 message 中提取内容
                            msg = event.message
                            if hasattr(msg, "content"):
                                for content in msg.content:
                                    if hasattr(content, "text"):
                                        content_parts.append(content.text)

                # 计算统计
                stats.finalize()
                stats.tokens = stats.r_tokens + stats.c_tokens

                # 构建最终响应内容
                final_content = "".join(content_parts) if content_parts else ""
                final_reasoning = "".join(reasoning_parts) if reasoning_parts else ""

                # 保存到历史
                if final_content:
                    self._messages.append(
                        {"role": "assistant", "content": final_content}
                    )

                # 发送统计事件
                stream.push(
                    AgentEvent(
                        type=AgentEventType.STATS,
                        stats={
                            "tokens": stats.tokens,
                            "r_tokens": stats.r_tokens,
                            "c_tokens": stats.c_tokens,
                            "elapsed": stats.elapsed,
                            "tps": stats.tps,
                            "ttft": stats.ttft,
                        },
                    )
                )

                # 发送结束事件
                stream.push(AgentEvent(type=AgentEventType.AGENT_END))

                # 构建最终结果
                result = ChatResult(
                    content=final_content,
                    messages=[{"role": "assistant", "content": final_content}],
                    model=model,
                )
                stream.end(result)

            except asyncio.CancelledError:
                # 处理取消
                final_content = "".join(content_parts) if content_parts else ""
                if final_content:
                    self._messages.append(
                        {"role": "assistant", "content": final_content}
                    )
                stream.push(
                    AgentEvent(
                        type=AgentEventType.ERROR,
                        error="Cancelled by user",
                    )
                )
                stream.end(
                    ChatResult(
                        content=final_content, messages=[], finish_reason="cancelled"
                    )
                )
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
        """列出可用模型."""
        provider = self._get_provider()
        if provider is None:
            return []
        try:
            return await provider.list_models()
        except Exception:
            return []


class LegacyDaemon(Daemon):
    """向后兼容的 Daemon - 支持旧的初始化方式."""

    def __init__(self, cfg):
        """从旧 Config 初始化.

        Args:
            cfg: 旧的 Config dataclass (包含 base_url, api_key, model, debug, history, reasoning_effort)
        """
        # 从 cfg 提取参数
        model = getattr(cfg, "model", None)
        history_limit = getattr(cfg, "history", 10)

        # 确定 provider_api
        provider_api = "openai"  # 默认
        base_url = getattr(cfg, "base_url", "")
        if base_url and "anthropic" in base_url.lower():
            provider_api = "anthropic"

        super().__init__(
            provider_api=provider_api,
            model=model,
            history_limit=history_limit,
        )

    @property
    def client(self):
        """向后兼容：返回 None (新架构不使用 client 属性)."""
        return None

    def clear(self) -> None:
        """向后兼容：clear -> clear_history."""
        return self.clear_history()
