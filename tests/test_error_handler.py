"""错误处理测试."""

import pytest
from tui_chatbot.error_handler import ErrorHandler, ErrorCategory, FriendlyError


class TestErrorHandler:
    """测试错误处理器."""

    def test_categorize_auth_error(self):
        """测试认证错误分类."""
        error = Exception("Invalid API key provided")
        category = ErrorHandler.categorize(error)
        assert category == ErrorCategory.AUTH

    def test_categorize_network_error(self):
        """测试网络错误分类."""
        error = Exception("Connection timeout")
        category = ErrorHandler.categorize(error)
        assert category == ErrorCategory.NETWORK

    def test_categorize_rate_limit(self):
        """测试限流错误分类."""
        error = Exception("Rate limit exceeded: 429")
        category = ErrorHandler.categorize(error)
        assert category == ErrorCategory.RATE_LIMIT

    def test_categorize_model_error(self):
        """测试模型错误分类."""
        error = Exception("Model not found")
        category = ErrorHandler.categorize(error)
        assert category == ErrorCategory.MODEL

    def test_categorize_content_error(self):
        """测试内容错误分类."""
        error = Exception("Content policy violation")
        category = ErrorHandler.categorize(error)
        assert category == ErrorCategory.CONTENT

    def test_categorize_unknown_error(self):
        """测试未知错误分类."""
        error = Exception("Some random error message")
        category = ErrorHandler.categorize(error)
        assert category == ErrorCategory.UNKNOWN

    def test_handle_returns_friendly_error(self):
        """测试错误处理返回友好信息."""
        error = Exception("Invalid API key")
        friendly = ErrorHandler.handle(error)

        assert isinstance(friendly, FriendlyError)
        assert friendly.category == ErrorCategory.AUTH
        assert "API 密钥无效" in friendly.title
        assert len(friendly.suggestion) > 0

    def test_format_output(self):
        """测试格式化输出包含所有部分."""
        error = Exception("Connection failed")
        output = ErrorHandler.format(error)

        assert "⚠️" in output
        assert "网络连接失败" in output
        assert "💡" in output

    def test_format_with_long_message(self):
        """测试长错误消息被截断."""
        long_message = "A" * 200
        error = Exception(long_message)
        output = ErrorHandler.format(error)

        # 输出中应该只包含前100个字符
        assert len(output) < 500

    def test_get_message_compatibility(self):
        """测试 get_message 兼容性接口."""
        error = Exception("Authentication failed")
        output = ErrorHandler.get_message(error)

        assert "API 密钥无效" in output
        assert "⚠️" in output
