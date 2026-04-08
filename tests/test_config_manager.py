"""配置管理器测试."""

import pytest
import tempfile
import json
from pathlib import Path

from tui_chatbot.config import ConfigManager, UserConfig


class TestConfigManager:
    """测试 ConfigManager"""

    @pytest.fixture
    def temp_config(self):
        """使用临时目录的配置管理器"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            yield ConfigManager(config_dir=tmp_path)

    def test_default_config(self, temp_config):
        """测试默认配置"""
        config = temp_config.get()

        assert config.default_model == "gpt-3.5-turbo"
        assert config.default_provider == "openai"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.theme == "default"
        assert config.auto_save is True

    def test_update_config(self, temp_config):
        """测试更新配置"""
        temp_config.update(temperature=0.5)

        assert temp_config.get().temperature == 0.5

    def test_update_multiple_fields(self, temp_config):
        """测试更新多个配置字段"""
        temp_config.update(
            temperature=0.3,
            theme="dark",
            max_tokens=2048,
        )

        config = temp_config.get()
        assert config.temperature == 0.3
        assert config.theme == "dark"
        assert config.max_tokens == 2048
        # 未修改的字段保持不变
        assert config.default_model == "gpt-3.5-turbo"

    def test_update_invalid_field_ignored(self, temp_config):
        """测试更新无效字段被忽略"""
        original_model = temp_config.get().default_model

        temp_config.update(nonexistent_field="value")

        # 无效字段不应影响配置
        assert temp_config.get().default_model == original_model

    def test_api_key_management(self, temp_config):
        """测试 API key 管理"""
        temp_config.set_api_key("openai", "sk-test123")

        assert temp_config.get_api_key("openai") == "sk-test123"

    def test_api_key_not_found(self, temp_config):
        """测试获取不存在的 API key"""
        result = temp_config.get_api_key("nonexistent")
        assert result is None

    def test_multiple_api_keys(self, temp_config):
        """测试多个 provider 的 API key"""
        temp_config.set_api_key("openai", "sk-openai")
        temp_config.set_api_key("anthropic", "sk-anthropic")
        temp_config.set_api_key("ollama", "sk-ollama")

        assert temp_config.get_api_key("openai") == "sk-openai"
        assert temp_config.get_api_key("anthropic") == "sk-anthropic"
        assert temp_config.get_api_key("ollama") == "sk-ollama"

    def test_update_api_key(self, temp_config):
        """测试更新已存在的 API key"""
        temp_config.set_api_key("openai", "sk-old")
        temp_config.set_api_key("openai", "sk-new")

        assert temp_config.get_api_key("openai") == "sk-new"

    def test_persistence_save_and_load(self):
        """测试配置持久化"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # 第一个实例创建并修改配置
            manager1 = ConfigManager(config_dir=tmp_path)
            manager1.update(theme="dark", temperature=0.5)
            manager1.set_api_key("openai", "sk-persist")

            # 第二个实例加载配置
            manager2 = ConfigManager(config_dir=tmp_path)
            config = manager2.get()

            assert config.theme == "dark"
            assert config.temperature == 0.5
            assert manager2.get_api_key("openai") == "sk-persist"

    def test_save_method(self, temp_config):
        """测试显式保存配置"""
        temp_config.update(theme="light")
        temp_config.save()

        # 验证文件内容
        config_file = temp_config._config_file
        assert config_file.exists()

        with open(config_file) as f:
            data = json.load(f)

        assert data["theme"] == "light"

    def test_load_corrupted_config(self, temp_config):
        """测试加载损坏的配置文件"""
        # 写入损坏的 JSON
        with open(temp_config._config_file, "w") as f:
            f.write("not valid json")

        # 重新加载应该返回默认配置
        config = temp_config._load()
        assert config.default_model == "gpt-3.5-turbo"

    def test_load_missing_file_returns_defaults(self, temp_config):
        """测试加载缺失的配置文件返回默认值"""
        # 确保文件不存在
        if temp_config._config_file.exists():
            temp_config._config_file.unlink()

        config = temp_config._load()
        assert config.default_model == "gpt-3.5-turbo"
        assert config.temperature == 0.7

    def test_config_dir_creation(self):
        """测试配置目录自动创建"""
        with tempfile.TemporaryDirectory() as tmp:
            new_dir = Path(tmp) / "new_config_dir"
            assert not new_dir.exists()

            ConfigManager(config_dir=new_dir)

            assert new_dir.exists()


class TestUserConfig:
    """测试 UserConfig 模型"""

    def test_default_values(self):
        """测试 UserConfig 默认值"""
        config = UserConfig()

        assert config.default_model == "gpt-3.5-turbo"
        assert config.default_provider == "openai"
        assert config.api_keys == {}
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.theme == "default"
        assert config.auto_save is True

    def test_custom_values(self):
        """测试 UserConfig 自定义值"""
        config = UserConfig(
            default_model="gpt-4",
            temperature=0.9,
            theme="light",
        )

        assert config.default_model == "gpt-4"
        assert config.temperature == 0.9
        assert config.theme == "light"

    def test_model_validation(self):
        """测试 UserConfig Pydantic 验证"""
        # 有效的配置
        data = {
            "default_model": "gpt-4",
            "temperature": 0.5,
            "api_keys": {"openai": "sk-test"},
        }
        config = UserConfig.model_validate(data)

        assert config.default_model == "gpt-4"
        assert config.temperature == 0.5
        assert config.api_keys["openai"] == "sk-test"

    def test_model_dump(self):
        """测试 UserConfig 序列化"""
        config = UserConfig(
            default_model="gpt-4",
            api_keys={"openai": "sk-test"},
        )
        data = config.model_dump()

        assert data["default_model"] == "gpt-4"
        assert data["api_keys"]["openai"] == "sk-test"
        assert "temperature" in data
