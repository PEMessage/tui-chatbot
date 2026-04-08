"""Provider 抽象基类."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from ..agent.types import AgentEvent, ChatResult
    from ..core.event_stream import EventStream


class ProviderConfig(BaseModel):
    """Provider 配置基类."""

    model_config = ConfigDict(frozen=True)

    api_key: str = Field(default="", description="API 密钥")
    base_url: str = Field(
        default="https://api.openai.com/v1", description="API 基础 URL"
    )
    model: str = Field(default="gpt-3.5-turbo", description="默认模型")
    timeout: Optional[float] = Field(default=60.0, description="请求超时时间")


class Provider(ABC):
    """LLM 提供商抽象基类.

    所有 LLM 提供商必须继承此类并实现抽象方法。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """提供商名称 (如 'openai', 'anthropic')."""
        pass

    @property
    @abstractmethod
    def api_type(self) -> str:
        """API 类型标识 (如 'openai-chat', 'anthropic-messages')."""
        pass

    @abstractmethod
    async def stream_chat(
        self,
        model: str,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        signal: Optional["AbortSignal"] = None,  # type: ignore  # noqa: F821
        temperature: Optional[float] = None,  # 新增
        max_tokens: Optional[int] = None,  # 新增
    ) -> "EventStream[AgentEvent, ChatResult]":
        """流式对话接口.

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
        pass

    @abstractmethod
    async def list_models(self) -> List[str]:
        """列出可用模型.

        Returns:
            List[str]: 模型 ID 列表
        """
        pass

    def to_dict(self) -> dict:
        """转换为字典表示."""
        return {
            "name": self.name,
            "api_type": self.api_type,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name} ({self.api_type})>"

    def __str__(self) -> str:
        return f"{self.name} ({self.api_type})"
