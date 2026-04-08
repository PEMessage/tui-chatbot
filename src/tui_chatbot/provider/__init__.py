"""Provider Registry 模块.

提供 LLM 提供商抽象基类、注册表和具体实现。

Usage:
    from tui_chatbot.provider import (
        Provider,
        ProviderConfig,
        ProviderRegistry,
        OpenAIProvider,
        OpenAIProviderConfig,
        LazyProvider,
        create_provider_from_env,
        register_default_providers,
    )

    # 注册提供商
    ProviderRegistry.register("openai", OpenAIProvider())

    # 获取提供商
    provider = ProviderRegistry.get("openai")

    # 流式对话
    stream = await provider.stream_chat(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
    async for event in stream:
        print(event)
"""

from .base import Provider, ProviderConfig
from .openai_provider import OpenAIProvider, OpenAIProviderConfig
from .registry import (
    LazyProvider,
    ProviderLoader,
    ProviderRegistry,
    create_provider_from_env,
    register_default_providers,
)

__all__ = [
    # 基类
    "Provider",
    "ProviderConfig",
    # 注册表
    "ProviderRegistry",
    "LazyProvider",
    "ProviderLoader",
    # OpenAI 实现
    "OpenAIProvider",
    "OpenAIProviderConfig",
    # 工具函数
    "create_provider_from_env",
    "register_default_providers",
]
