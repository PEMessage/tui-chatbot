"""Provider Registry 实现."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Awaitable, Callable, Dict, List, Optional, Type

from .base import Provider

if TYPE_CHECKING:
    from ..agent.types import AgentEvent, ChatResult
    from ..core.event_stream import EventStream


# 延迟加载器类型
ProviderLoader = Callable[[], Awaitable[Provider]]


class LazyProvider(Provider):
    """延迟加载的提供商包装器.

    在首次访问时才实际加载 Provider 实例，避免启动时导入 heavy 依赖。

    Usage:
        async def load_openai():
            from openai import AsyncOpenAI
            return OpenAIProvider(AsyncOpenAI())

        registry.register("openai", LazyProvider(load_openai))
    """

    def __init__(
        self,
        loader: ProviderLoader,
        name_hint: str = "lazy",
        api_type_hint: str = "lazy",
    ):
        """初始化延迟加载包装器.

        Args:
            loader: 异步加载函数，返回 Provider 实例
            name_hint: 加载前的名称提示
            api_type_hint: 加载前的 API 类型提示
        """
        self._loader = loader
        self._provider: Optional[Provider] = None
        self._loading: Optional[asyncio.Task[Provider]] = None
        self._name_hint = name_hint
        self._api_type_hint = api_type_hint

    async def _ensure_loaded(self) -> Provider:
        """确保 Provider 已加载 (线程安全)."""
        if self._provider is not None:
            return self._provider

        # 防止并发加载
        if self._loading is not None:
            return await self._loading

        async def _load():
            provider = await self._loader()
            self._provider = provider
            self._loading = None
            return provider

        self._loading = asyncio.create_task(_load())
        return await self._loading

    @property
    def name(self) -> str:
        """提供商名称 (加载后返回实际名称)."""
        if self._provider is not None:
            return self._provider.name
        return self._name_hint

    @property
    def api_type(self) -> str:
        """API 类型 (加载后返回实际类型)."""
        if self._provider is not None:
            return self._provider.api_type
        return self._api_type_hint

    async def stream_chat(
        self,
        model: str,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        signal: Optional["AbortSignal"] = None,  # type: ignore  # noqa: F821
    ) -> "EventStream[AgentEvent, ChatResult]":
        """流式对话 (代理到实际 Provider)."""
        provider = await self._ensure_loaded()
        return await provider.stream_chat(model, messages, tools, signal)

    async def list_models(self) -> List[str]:
        """列出模型 (代理到实际 Provider)."""
        provider = await self._ensure_loaded()
        return await provider.list_models()

    def to_dict(self) -> dict:
        """转换为字典表示."""
        if self._provider is not None:
            return self._provider.to_dict()
        return {
            "name": self.name,
            "api_type": self.api_type,
            "status": "not_loaded",
        }


class ProviderRegistry:
    """Provider 注册表 - 运行时动态注册 LLM 提供商.

    支持类级别的全局注册表和实例级别的注册表。
    推荐使用类方法访问全局注册表。

    Usage:
        # 注册提供商
        ProviderRegistry.register("openai", OpenAIProvider(client))

        # 延迟注册
        ProviderRegistry.register("openai", LazyProvider(load_openai))

        # 获取提供商
        provider = ProviderRegistry.get("openai")

        # 列出所有已注册
        apis = ProviderRegistry.list()
    """

    # 类级别存储: api_type -> Provider
    _providers: Dict[str, Provider] = {}

    @classmethod
    def register(cls, api_type: str, provider: Provider) -> None:
        """注册提供商.

        Args:
            api_type: API 类型标识 (如 'openai-chat', 'anthropic-messages')
            provider: Provider 实例或 LazyProvider 包装器

        Raises:
            TypeError: 如果 provider 不是 Provider 实例
        """
        if not isinstance(provider, Provider):
            raise TypeError(
                f"Provider must be instance of Provider, got {type(provider)}"
            )
        cls._providers[api_type] = provider

    @classmethod
    def register_lazy(
        cls,
        api_type: str,
        loader: ProviderLoader,
        name_hint: str = "lazy",
    ) -> None:
        """注册延迟加载的提供商.

        Args:
            api_type: API 类型标识
            loader: 异步加载函数
            name_hint: 加载前的名称提示
        """
        lazy_provider = LazyProvider(
            loader, name_hint=name_hint, api_type_hint=api_type
        )
        cls.register(api_type, lazy_provider)

    @classmethod
    def get(cls, api_type: str) -> Optional[Provider]:
        """获取提供商.

        Args:
            api_type: API 类型标识

        Returns:
            Provider 实例，如果未注册返回 None
        """
        return cls._providers.get(api_type)

    @classmethod
    def get_or_raise(cls, api_type: str) -> Provider:
        """获取提供商，如果不存在则抛出异常.

        Args:
            api_type: API 类型标识

        Returns:
            Provider 实例

        Raises:
            KeyError: 如果未注册该 API 类型
        """
        provider = cls.get(api_type)
        if provider is None:
            raise KeyError(f"Provider not registered: {api_type}")
        return provider

    @classmethod
    def list(cls) -> List[str]:
        """列出所有已注册的 API 类型.

        Returns:
            List[str]: API 类型标识列表
        """
        return list(cls._providers.keys())

    @classmethod
    def list_providers(cls) -> List[Provider]:
        """列出所有已注册的 Provider 实例.

        Returns:
            List[Provider]: Provider 实例列表
        """
        return list(cls._providers.values())

    @classmethod
    def clear(cls) -> None:
        """清空注册表 (主要用于测试)."""
        cls._providers.clear()

    @classmethod
    def unregister(cls, api_type: str) -> bool:
        """注销提供商.

        Args:
            api_type: API 类型标识

        Returns:
            bool: 是否成功移除
        """
        if api_type in cls._providers:
            del cls._providers[api_type]
            return True
        return False

    @classmethod
    def is_registered(cls, api_type: str) -> bool:
        """检查是否已注册.

        Args:
            api_type: API 类型标识

        Returns:
            bool: 是否已注册
        """
        return api_type in cls._providers

    @classmethod
    def get_info(cls) -> Dict[str, dict]:
        """获取所有注册信息.

        Returns:
            Dict[str, dict]: api_type -> provider info dict
        """
        return {
            api_type: provider.to_dict()
            for api_type, provider in cls._providers.items()
        }


def create_provider_from_env(api_type: str = "openai") -> Optional[Provider]:
    """从环境变量创建 Provider.

    根据 api_type 自动识别并创建对应的 Provider。

    Args:
        api_type: API 类型 (目前仅支持 'openai')

    Returns:
        Provider 实例，如果环境变量未配置返回 None
    """
    import os

    if api_type in ("openai", "openai-chat"):
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

        if not api_key:
            return None

        # 延迟导入避免循环依赖
        from openai import AsyncOpenAI

        from .openai_provider import OpenAIProvider, OpenAIProviderConfig

        config = OpenAIProviderConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        return OpenAIProvider(client=client, config=config)

    return None


def register_default_providers() -> None:
    """注册默认提供商 (从环境变量)."""
    # 尝试注册 OpenAI
    openai_provider = create_provider_from_env("openai")
    if openai_provider:
        ProviderRegistry.register("openai", openai_provider)
        ProviderRegistry.register("openai-chat", openai_provider)

    # 尝试注册 Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        from anthropic import AsyncAnthropic

        from .anthropic_provider import AnthropicProvider, AnthropicProviderConfig

        config = AnthropicProviderConfig()
        client = AsyncAnthropic(api_key=anthropic_key)
        ProviderRegistry.register("anthropic", AnthropicProvider(client, config))
