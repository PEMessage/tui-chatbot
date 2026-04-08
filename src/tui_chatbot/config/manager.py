"""配置管理模块 - 负责保存和加载用户配置"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class UserConfig(BaseModel):
    """用户配置"""

    default_model: str = "gpt-3.5-turbo"
    default_provider: str = "openai"
    api_keys: Dict[str, str] = Field(default_factory=dict)
    temperature: float = 0.7
    max_tokens: int = 4096
    theme: str = "default"
    auto_save: bool = True


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            config_dir = Path.home() / ".config" / "tui-chatbot"
        self._config_dir = config_dir
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._config_file = self._config_dir / "config.json"
        self._config = self._load()

    def _load(self) -> UserConfig:
        """加载配置"""
        if not self._config_file.exists():
            return UserConfig()
        try:
            with open(self._config_file) as f:
                data = json.load(f)
            return UserConfig.model_validate(data)
        except Exception:
            return UserConfig()

    def save(self) -> None:
        """保存配置"""
        with open(self._config_file, "w") as f:
            json.dump(self._config.model_dump(), f, indent=2)

    def get(self) -> UserConfig:
        """获取当前配置"""
        return self._config

    def update(self, **kwargs) -> None:
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        if self._config.auto_save:
            self.save()

    def get_api_key(self, provider: str) -> Optional[str]:
        """获取指定 provider 的 API key"""
        return self._config.api_keys.get(provider)

    def set_api_key(self, provider: str, key: str) -> None:
        """设置 API key"""
        self._config.api_keys[provider] = key
        if self._config.auto_save:
            self.save()


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
