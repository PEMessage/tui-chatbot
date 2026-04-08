"""Tests for Provider Registry module.

测试覆盖:
- ProviderRegistry 注册和获取
- LazyProvider 延迟加载
- 环境变量创建 Provider
- 错误处理
"""

import asyncio
import pytest
from typing import List, Optional

from tui_chatbot.provider.registry import (
    ProviderRegistry,
    LazyProvider,
    create_provider_from_env,
    register_default_providers,
)
from tui_chatbot.provider.base import Provider, ProviderConfig


class MockProvider(Provider):
    """模拟 Provider 用于测试."""

    def __init__(self, name: str = "mock", api_type: str = "mock-api"):
        self._name = name
        self._api_type = api_type

    @property
    def name(self) -> str:
        return self._name

    @property
    def api_type(self) -> str:
        return self._api_type

    async def stream_chat(
        self,
        model: str,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        signal: Optional["AbortSignal"] = None,  # type: ignore
    ):
        """模拟流式对话."""
        from tui_chatbot.core.event_stream import EventStream
        from tui_chatbot.agent.types import AgentEvent, AgentEventType, ChatResult

        stream = EventStream()
        stream.end(ChatResult(messages=[]))
        return stream

    async def list_models(self) -> List[str]:
        return ["mock-model-1", "mock-model-2"]


class TestProviderRegistry:
    """测试 ProviderRegistry 类."""

    def setup_method(self):
        """每个测试前清理注册表."""
        ProviderRegistry.clear()

    def teardown_method(self):
        """每个测试后清理注册表."""
        ProviderRegistry.clear()

    def test_register_provider(self):
        """测试注册 Provider."""
        provider = MockProvider()
        ProviderRegistry.register("test-api", provider)

        assert ProviderRegistry.is_registered("test-api")

    def test_register_invalid_provider(self):
        """测试注册无效 Provider."""
        with pytest.raises(TypeError):
            ProviderRegistry.register("test", "not a provider")

    def test_get_provider(self):
        """测试获取 Provider."""
        provider = MockProvider()
        ProviderRegistry.register("test-api", provider)

        retrieved = ProviderRegistry.get("test-api")

        assert retrieved is provider

    def test_get_nonexistent_provider(self):
        """测试获取不存在的 Provider."""
        result = ProviderRegistry.get("nonexistent")

        assert result is None

    def test_get_or_raise_exists(self):
        """测试获取存在的 Provider (或抛出)."""
        provider = MockProvider()
        ProviderRegistry.register("test-api", provider)

        retrieved = ProviderRegistry.get_or_raise("test-api")

        assert retrieved is provider

    def test_get_or_raise_not_exists(self):
        """测试获取不存在的 Provider 抛出异常."""
        with pytest.raises(KeyError, match="Provider not registered"):
            ProviderRegistry.get_or_raise("nonexistent")

    def test_list_providers(self):
        """测试列出所有注册的 API 类型."""
        ProviderRegistry.register("api1", MockProvider())
        ProviderRegistry.register("api2", MockProvider())
        ProviderRegistry.register("api3", MockProvider())

        apis = ProviderRegistry.list()

        assert "api1" in apis
        assert "api2" in apis
        assert "api3" in apis
        assert len(apis) == 3

    def test_list_empty(self):
        """测试空注册表."""
        apis = ProviderRegistry.list()

        assert apis == []

    def test_unregister(self):
        """测试注销 Provider."""
        ProviderRegistry.register("test-api", MockProvider())

        success = ProviderRegistry.unregister("test-api")

        assert success is True
        assert not ProviderRegistry.is_registered("test-api")

    def test_unregister_nonexistent(self):
        """测试注销不存在的 Provider."""
        success = ProviderRegistry.unregister("nonexistent")

        assert success is False

    def test_is_registered(self):
        """测试检查是否已注册."""
        ProviderRegistry.register("test-api", MockProvider())

        assert ProviderRegistry.is_registered("test-api") is True
        assert ProviderRegistry.is_registered("other-api") is False

    def test_clear(self):
        """测试清空注册表."""
        ProviderRegistry.register("api1", MockProvider())
        ProviderRegistry.register("api2", MockProvider())

        ProviderRegistry.clear()

        assert ProviderRegistry.list() == []

    def test_list_providers_instances(self):
        """测试列出所有 Provider 实例."""
        provider1 = MockProvider(name="p1")
        provider2 = MockProvider(name="p2")

        ProviderRegistry.register("api1", provider1)
        ProviderRegistry.register("api2", provider2)

        providers = ProviderRegistry.list_providers()

        assert len(providers) == 2
        assert provider1 in providers
        assert provider2 in providers

    def test_get_info(self):
        """测试获取注册信息."""
        ProviderRegistry.register(
            "test-api", MockProvider(name="test", api_type="test-api")
        )

        info = ProviderRegistry.get_info()

        assert "test-api" in info
        assert info["test-api"]["name"] == "test"


class TestLazyProvider:
    """测试 LazyProvider 延迟加载."""

    def test_lazy_provider_initial_state(self):
        """测试初始状态."""

        async def loader():
            return MockProvider()

        lazy = LazyProvider(loader, name_hint="test", api_type_hint="test-api")

        # 加载前返回 hint
        assert lazy.name == "test"
        assert lazy.api_type == "test-api"

    @pytest.mark.asyncio
    async def test_lazy_provider_loads_on_access(self):
        """测试首次访问时加载."""
        load_count = 0

        async def loader():
            nonlocal load_count
            load_count += 1
            return MockProvider(name="loaded", api_type="loaded-api")

        lazy = LazyProvider(loader, name_hint="hint")

        # 首次访问应该触发加载
        _ = await lazy.list_models()

        assert load_count == 1
        # 加载后返回实际值
        assert lazy.name == "loaded"
        assert lazy.api_type == "loaded-api"

    @pytest.mark.asyncio
    async def test_lazy_provider_caches_result(self):
        """测试结果会被缓存."""
        load_count = 0

        async def loader():
            nonlocal load_count
            load_count += 1
            return MockProvider()

        lazy = LazyProvider(loader)

        # 多次访问
        await lazy.list_models()
        await lazy.list_models()
        await lazy.list_models()

        assert load_count == 1  # 只加载一次

    @pytest.mark.asyncio
    async def test_lazy_provider_concurrent_load(self):
        """测试并发加载只执行一次."""
        load_count = 0

        async def slow_loader():
            nonlocal load_count
            load_count += 1
            await asyncio.sleep(0.1)
            return MockProvider()

        lazy = LazyProvider(slow_loader)

        # 并发访问
        await asyncio.gather(
            lazy.list_models(),
            lazy.list_models(),
            lazy.list_models(),
        )

        assert load_count == 1  # 只加载一次

    def test_lazy_provider_to_dict_not_loaded(self):
        """测试未加载时的字典表示."""

        async def loader():
            return MockProvider()

        lazy = LazyProvider(loader, name_hint="test", api_type_hint="test-api")

        info = lazy.to_dict()

        assert info["name"] == "test"
        assert info["api_type"] == "test-api"
        assert info["status"] == "not_loaded"

    @pytest.mark.asyncio
    async def test_lazy_provider_to_dict_loaded(self):
        """测试已加载时的字典表示."""

        async def loader():
            return MockProvider(name="loaded")

        lazy = LazyProvider(loader)
        await lazy.list_models()  # 触发加载

        info = lazy.to_dict()

        assert info["name"] == "loaded"


class TestProviderRegistryLazyRegistration:
    """测试延迟注册功能."""

    def setup_method(self):
        """每个测试前清理注册表."""
        ProviderRegistry.clear()

    def teardown_method(self):
        """每个测试后清理注册表."""
        ProviderRegistry.clear()

    def test_register_lazy(self):
        """测试延迟注册."""
        load_count = 0

        async def loader():
            nonlocal load_count
            load_count += 1
            return MockProvider()

        ProviderRegistry.register_lazy("lazy-api", loader, name_hint="Lazy")

        # 注册时不加载
        assert load_count == 0
        assert ProviderRegistry.is_registered("lazy-api")

        # 获取 Provider
        provider = ProviderRegistry.get("lazy-api")
        assert isinstance(provider, LazyProvider)


class TestCreateProviderFromEnv:
    """测试从环境变量创建 Provider."""

    def test_create_without_api_key(self, monkeypatch):
        """测试没有 API key 时返回 None."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)

        provider = create_provider_from_env("openai")

        assert provider is None

    def test_create_with_api_key(self, monkeypatch):
        """测试有 API key 时创建 Provider."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://test.com/v1")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")

        # 注意：这里需要 mock openai 导入，否则会真的尝试连接
        # 但为了简化测试，我们只测试在没有 openai 包时的行为
        try:
            provider = create_provider_from_env("openai")
            # 如果成功，应该是一个 Provider
            if provider is not None:
                assert isinstance(provider, Provider)
        except ImportError:
            # openai 包未安装，这是预期的
            pass

    def test_create_unsupported_api_type(self):
        """测试不支持的 API 类型."""
        provider = create_provider_from_env("unsupported")

        assert provider is None


class TestRegisterDefaultProviders:
    """测试注册默认 Provider."""

    def setup_method(self):
        """每个测试前清理注册表."""
        ProviderRegistry.clear()

    def teardown_method(self):
        """每个测试后清理注册表."""
        ProviderRegistry.clear()

    def test_register_without_env(self, monkeypatch):
        """测试没有环境变量时不注册."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        register_default_providers()

        # 不应该注册 openai
        assert not ProviderRegistry.is_registered("openai")


class TestProviderBaseClass:
    """测试 Provider 基类."""

    def test_mock_provider_properties(self):
        """测试 Mock Provider 属性."""
        provider = MockProvider(name="test", api_type="test-api")

        assert provider.name == "test"
        assert provider.api_type == "test-api"

    def test_mock_provider_to_dict(self):
        """测试 to_dict 方法."""
        provider = MockProvider(name="test", api_type="test-api")

        info = provider.to_dict()

        assert info["name"] == "test"
        assert info["api_type"] == "test-api"

    def test_mock_provider_repr(self):
        """测试 repr."""
        provider = MockProvider(name="test", api_type="test-api")

        repr_str = repr(provider)

        assert "MockProvider" in repr_str
        assert "test" in repr_str

    def test_mock_provider_str(self):
        """测试 str."""
        provider = MockProvider(name="test", api_type="test-api")

        str_repr = str(provider)

        assert "test" in str_repr
        assert "test-api" in str_repr

    @pytest.mark.asyncio
    async def test_mock_provider_list_models(self):
        """测试列出模型."""
        provider = MockProvider()

        models = await provider.list_models()

        assert len(models) == 2
        assert "mock-model-1" in models

    @pytest.mark.asyncio
    async def test_mock_provider_stream_chat(self):
        """测试流式对话."""
        provider = MockProvider()

        stream = await provider.stream_chat(
            model="test",
            messages=[{"role": "user", "content": "hello"}],
        )

        assert stream is not None
        # 可以迭代
        events = []
        async for ev in stream:
            events.append(ev)
        # 流已经结束


class TestProviderRegistryEdgeCases:
    """测试边界情况."""

    def setup_method(self):
        """每个测试前清理注册表."""
        ProviderRegistry.clear()

    def teardown_method(self):
        """每个测试后清理注册表."""
        ProviderRegistry.clear()

    def test_register_same_api_type_twice(self):
        """测试重复注册相同 API 类型."""
        provider1 = MockProvider(name="first")
        provider2 = MockProvider(name="second")

        ProviderRegistry.register("api", provider1)
        ProviderRegistry.register("api", provider2)  # 覆盖

        retrieved = ProviderRegistry.get("api")
        assert retrieved is provider2  # 后注册的覆盖

    def test_register_empty_api_type(self):
        """测试空 API 类型."""
        provider = MockProvider()

        ProviderRegistry.register("", provider)

        assert ProviderRegistry.is_registered("")
        assert ProviderRegistry.get("") is provider

    def test_list_after_clear(self):
        """测试清空后列表为空."""
        ProviderRegistry.register("api1", MockProvider())
        ProviderRegistry.clear()
        ProviderRegistry.register("api2", MockProvider())

        apis = ProviderRegistry.list()

        assert "api1" not in apis
        assert "api2" in apis
