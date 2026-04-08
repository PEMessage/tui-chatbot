"""Provider 错误处理 - 友好的错误提示."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict


class ErrorCategory(Enum):
    """错误分类."""

    AUTH = auto()  # 认证错误 (API Key)
    NETWORK = auto()  # 网络错误
    RATE_LIMIT = auto()  # 限流错误
    MODEL = auto()  # 模型错误 (不存在、不可用)
    CONTENT = auto()  # 内容错误 (过长、敏感)
    UNKNOWN = auto()  # 未知错误


@dataclass
class FriendlyError:
    """友好的错误信息."""

    category: ErrorCategory
    original: str
    title: str
    message: str
    suggestion: str


class ErrorHandler:
    """错误处理器 - 将 Provider 错误转换为友好提示."""

    # 错误模式映射
    ERROR_PATTERNS: Dict[ErrorCategory, list[str]] = {
        ErrorCategory.AUTH: [
            "invalid api key",
            "incorrect api key",
            "unauthorized",
            "authentication",
            "401",
            "api key",
        ],
        ErrorCategory.NETWORK: [
            "connection",
            "timeout",
            "unable to connect",
            "network",
            "dns",
        ],
        ErrorCategory.RATE_LIMIT: [
            "rate limit",
            "too many requests",
            "429",
            "quota exceeded",
        ],
        ErrorCategory.MODEL: [
            "model not found",
            "model does not exist",
            "invalid model",
            "not found",
        ],
        ErrorCategory.CONTENT: [
            "content filter",
            "too long",
            "maximum context",
            "content",
            "moderation",
            "policy",
        ],
    }

    # 友好消息模板
    FRIENDLY_MESSAGES: Dict[ErrorCategory, tuple[str, str]] = {
        ErrorCategory.AUTH: (
            "API 密钥无效",
            "请检查 OPENAI_API_KEY 环境变量或 --api-key 参数是否正确设置。",
        ),
        ErrorCategory.NETWORK: (
            "网络连接失败",
            "请检查网络连接，或尝试使用 --base-url 指定其他 API 地址。",
        ),
        ErrorCategory.RATE_LIMIT: ("请求过于频繁", "API 限流中，请稍等片刻后重试。"),
        ErrorCategory.MODEL: ("模型不可用", "请使用 /model 命令查看可用模型列表。"),
        ErrorCategory.CONTENT: (
            "内容处理错误",
            "消息可能过长或包含敏感内容，请尝试缩短或修改后重试。",
        ),
        ErrorCategory.UNKNOWN: (
            "发生错误",
            "请检查日志或重试。如果问题持续，请报告问题。",
        ),
    }

    @classmethod
    def categorize(cls, error: Exception) -> ErrorCategory:
        """对错误进行分类.

        Args:
            error: 原始错误

        Returns:
            ErrorCategory: 错误分类
        """
        error_str = str(error).lower()

        for category, patterns in cls.ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern in error_str:
                    return category

        return ErrorCategory.UNKNOWN

    @classmethod
    def handle(cls, error: Exception) -> FriendlyError:
        """处理错误并返回友好信息.

        Args:
            error: 原始错误

        Returns:
            FriendlyError: 友好的错误信息
        """
        category = cls.categorize(error)
        title, suggestion = cls.FRIENDLY_MESSAGES[category]

        return FriendlyError(
            category=category,
            original=str(error),
            title=title,
            message=str(error),
            suggestion=suggestion,
        )

    @classmethod
    def format(cls, error: Exception) -> str:
        """格式化错误为显示字符串.

        Args:
            error: 原始错误

        Returns:
            str: 格式化后的错误信息
        """
        friendly = cls.handle(error)

        lines = [
            f"\n⚠️  {friendly.title}",
            f"   {friendly.message[:100]}",
            "",
            f"💡 {friendly.suggestion}",
        ]

        return "\n".join(lines)

    @classmethod
    def get_message(cls, error: Exception) -> str:
        """获取友好的错误消息 (兼容旧接口).

        Args:
            error: 原始错误

        Returns:
            str: 友好的错误消息
        """
        return cls.format(error)
