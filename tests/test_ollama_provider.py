"""OllamaProvider 单元测试."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tui_chatbot.agent.types import AgentEventType
from tui_chatbot.provider.ollama_provider import OllamaProvider, OllamaProviderConfig


def async_iter_lines(lines):
    """创建异步迭代器模拟 aiter_lines."""

    async def _aiter_lines():
        for line in lines:
            yield line

    return _aiter_lines()


class TestOllamaProviderConfig:
    """测试 OllamaProviderConfig."""

    def test_default_config(self):
        """测试默认配置."""
        config = OllamaProviderConfig()
        assert config.model == "llama3.2"
        assert config.base_url == "http://localhost:11434"
        assert config.temperature is None
        assert config.max_tokens is None
        assert config.top_p is None

    def test_custom_config(self):
        """测试自定义配置."""
        config = OllamaProviderConfig(
            model="mistral",
            base_url="http://192.168.1.100:11434",
            temperature=0.8,
            max_tokens=2048,
            top_p=0.9,
        )
        assert config.model == "mistral"
        assert config.base_url == "http://192.168.1.100:11434"
        assert config.temperature == 0.8
        assert config.max_tokens == 2048
        assert config.top_p == 0.9


class TestOllamaProvider:
    """测试 OllamaProvider."""

    def test_provider_properties(self):
        """测试 Provider 基本属性."""
        provider = OllamaProvider()
        assert provider.name == "ollama"
        assert provider.api_type == "ollama"

    def test_provider_with_custom_config(self):
        """测试使用自定义配置."""
        config = OllamaProviderConfig(model="codellama")
        provider = OllamaProvider(config)
        assert provider._config.model == "codellama"

    def test_provider_to_dict(self):
        """测试 to_dict 方法."""
        provider = OllamaProvider()
        info = provider.to_dict()
        assert info["name"] == "ollama"
        assert info["api_type"] == "ollama"


class TestOllamaProviderListModels:
    """测试 list_models 方法."""

    async def test_list_models_success(self):
        """测试成功获取模型列表."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "mistral:latest"},
                {"name": "codellama:7b"},
            ]
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = OllamaProvider()
        provider._client = mock_client

        models = await provider.list_models()
        assert len(models) == 3
        assert "llama3.2:latest" in models
        assert "mistral:latest" in models
        assert "codellama:7b" in models

    async def test_list_models_empty(self):
        """测试空模型列表."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": []}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        provider = OllamaProvider()
        provider._client = mock_client

        models = await provider.list_models()
        assert models == []

    async def test_list_models_error(self):
        """测试获取模型列表失败."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))

        provider = OllamaProvider()
        provider._client = mock_client

        models = await provider.list_models()
        assert models == []


class TestOllamaProviderStreamChat:
    """测试 stream_chat 方法."""

    async def test_stream_chat_basic(self):
        """测试基本流式对话."""
        # 模拟 Ollama 流式响应
        stream_lines = [
            json.dumps(
                {"message": {"role": "assistant", "content": "Hello"}, "done": False}
            ),
            json.dumps(
                {"message": {"role": "assistant", "content": " World"}, "done": False}
            ),
            json.dumps(
                {"message": {"role": "assistant", "content": "!"}, "done": True}
            ),
        ]

        # 创建 mock response
        mock_response = MagicMock()
        mock_response.aiter_lines = MagicMock(
            return_value=async_iter_lines(stream_lines)
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)

        provider = OllamaProvider()
        provider._client = mock_client

        stream = await provider.stream_chat(
            model="llama3.2",
            messages=[{"role": "user", "content": "Hi"}],
        )

        events = []
        async for event in stream:
            events.append(event)

        result = await stream.result()

        # 验证事件序列
        assert (
            len(events) >= 4
        )  # MESSAGE_START, MESSAGE_UPDATE x 3, MESSAGE_END, TURN_END
        assert events[0].type == AgentEventType.MESSAGE_START
        assert events[-2].type == AgentEventType.MESSAGE_END
        assert events[-1].type == AgentEventType.TURN_END

        # 验证结果
        assert result is not None
        assert len(result.messages) == 1
        assert result.finish_reason == "stop"

    async def test_stream_chat_with_temperature(self):
        """测试带 temperature 参数的流式对话."""
        stream_lines = [
            json.dumps({"message": {"content": "Test"}, "done": True}),
        ]

        mock_response = MagicMock()
        mock_response.aiter_lines = MagicMock(
            return_value=async_iter_lines(stream_lines)
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)

        provider = OllamaProvider()
        provider._client = mock_client

        stream = await provider.stream_chat(
            model="llama3.2",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
        )

        # 等待流式处理完成
        await stream.result()

        # 验证调用参数
        call_args = mock_client.stream.call_args
        assert call_args is not None
        _, kwargs = call_args
        payload = kwargs.get("json", {})
        assert payload["options"]["temperature"] == 0.5

    async def test_stream_chat_with_max_tokens(self):
        """测试带 max_tokens 参数的流式对话."""
        stream_lines = [
            json.dumps({"message": {"content": "Test"}, "done": True}),
        ]

        mock_response = MagicMock()
        mock_response.aiter_lines = MagicMock(
            return_value=async_iter_lines(stream_lines)
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)

        provider = OllamaProvider()
        provider._client = mock_client

        stream = await provider.stream_chat(
            model="llama3.2",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=512,
        )

        # 等待流式处理完成
        await stream.result()

        # 验证调用参数
        call_args = mock_client.stream.call_args
        assert call_args is not None
        _, kwargs = call_args
        payload = kwargs.get("json", {})
        assert payload["options"]["num_predict"] == 512

    async def test_stream_chat_uses_config_values(self):
        """测试使用 config 中的默认值."""
        stream_lines = [
            json.dumps({"message": {"content": "Test"}, "done": True}),
        ]

        mock_response = MagicMock()
        mock_response.aiter_lines = MagicMock(
            return_value=async_iter_lines(stream_lines)
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)

        config = OllamaProviderConfig(temperature=0.7, max_tokens=1024)
        provider = OllamaProvider(config)
        provider._client = mock_client

        stream = await provider.stream_chat(
            model="llama3.2",
            messages=[{"role": "user", "content": "Hi"}],
        )

        # 等待流式处理完成
        await stream.result()

        # 验证调用参数使用了 config 中的值
        call_args = mock_client.stream.call_args
        assert call_args is not None
        _, kwargs = call_args
        payload = kwargs.get("json", {})
        assert payload["options"]["temperature"] == 0.7
        assert payload["options"]["num_predict"] == 1024

    async def test_stream_chat_empty_response(self):
        """测试空响应情况."""
        stream_lines = [
            json.dumps({"message": {"content": ""}, "done": True}),
        ]

        mock_response = MagicMock()
        mock_response.aiter_lines = MagicMock(
            return_value=async_iter_lines(stream_lines)
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)

        provider = OllamaProvider()
        provider._client = mock_client

        stream = await provider.stream_chat(
            model="llama3.2",
            messages=[{"role": "user", "content": "Hi"}],
        )

        result = await stream.result()
        assert result is not None
        assert len(result.messages) == 1


class TestOllamaProviderClose:
    """测试 close 方法."""

    async def test_close_client(self):
        """测试关闭客户端."""
        mock_client = AsyncMock()

        provider = OllamaProvider()
        provider._client = mock_client

        await provider.close()

        mock_client.aclose.assert_called_once()
        assert provider._client is None
