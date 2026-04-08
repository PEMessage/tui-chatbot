"""流式指示器测试."""

import time
import pytest
from tui_chatbot.indicator import StreamingIndicator


class TestStreamingIndicator:
    """测试流式响应指示器."""

    def test_init_default_values(self):
        """测试默认初始化值."""
        indicator = StreamingIndicator()

        assert indicator.token_count == 0
        assert indicator.first_token_time is None
        assert indicator._active is False

    def test_start_initializes_state(self):
        """测试 start 方法初始化状态."""
        indicator = StreamingIndicator()
        indicator.start()

        assert indicator._active is True
        assert indicator.token_count == 0
        assert indicator.first_token_time is None

        indicator.stop()

    def test_on_first_token_records_time(self):
        """测试 on_first_token 记录时间."""
        indicator = StreamingIndicator()

        assert indicator.first_token_time is None

        indicator.on_first_token()

        assert indicator.first_token_time is not None
        assert indicator.first_token_time > 0

    def test_on_token_increments_count(self):
        """测试 on_token 增加计数."""
        indicator = StreamingIndicator()
        indicator.start()

        indicator.on_token()
        assert indicator.token_count == 1

        indicator.on_token()
        assert indicator.token_count == 2

        indicator.stop()

    def test_on_token_records_first_token_time(self):
        """测试 on_token 自动记录首 token 时间."""
        indicator = StreamingIndicator()

        assert indicator.first_token_time is None

        indicator.on_token()

        assert indicator.first_token_time is not None

    def test_update_increments_tokens(self):
        """测试 update 方法增加 token."""
        indicator = StreamingIndicator()
        indicator.start()

        indicator.update(tokens=5)
        assert indicator.token_count == 5

        indicator.update(tokens=3)
        assert indicator.token_count == 8

        indicator.stop()

    def test_update_when_inactive_does_nothing(self):
        """测试非活跃状态 update 不生效."""
        indicator = StreamingIndicator()
        # 不调用 start()

        indicator.update(tokens=5)
        # 由于 _active 为 False，token_count 不应改变
        assert indicator.token_count == 0

    def test_ttft_before_first_token(self):
        """测试首 token 前的 TTFT."""
        indicator = StreamingIndicator()
        time.sleep(0.1)

        ttft = indicator.ttft

        assert ttft >= 0.1

    def test_ttft_after_first_token(self):
        """测试首 token 后的 TTFT."""
        indicator = StreamingIndicator()
        time.sleep(0.05)
        indicator.on_first_token()
        time.sleep(0.05)

        ttft = indicator.ttft

        assert ttft >= 0.05
        assert ttft < 0.2  # 应该小于总时间

    def test_tps_before_first_token(self):
        """测试首 token 前的 TPS 为 0."""
        indicator = StreamingIndicator()

        assert indicator.tps == 0.0

    def test_tps_after_tokens(self):
        """测试收到 token 后的 TPS."""
        indicator = StreamingIndicator()
        indicator.start()

        time.sleep(0.05)
        indicator.on_token()
        time.sleep(0.05)
        indicator.on_token()

        tps = indicator.tps

        assert tps > 0
        indicator.stop()

    def test_render_thinking_state(self):
        """测试思考状态的渲染."""
        indicator = StreamingIndicator()

        output = indicator.render()

        assert "Thinking" in output or "🤔" in output
        assert "s)" in output  # 时间单位

    def test_render_generating_state(self):
        """测试生成状态的渲染."""
        indicator = StreamingIndicator()
        indicator.on_first_token()
        indicator.token_count = 10

        output = indicator.render()

        assert "Generating" in output or "✍️" in output
        assert "10 tokens" in output
        assert "tps" in output

    def test_stop_sets_inactive(self):
        """测试 stop 方法设置非活跃状态."""
        indicator = StreamingIndicator()
        indicator.start()

        assert indicator._active is True

        indicator.stop()

        assert indicator._active is False

    def test_multiple_start_resets_state(self):
        """测试多次调用 start 重置状态."""
        indicator = StreamingIndicator()
        indicator.start()
        indicator.on_token()
        indicator.on_token()

        assert indicator.token_count == 2

        indicator.start()  # 重新开始

        assert indicator.token_count == 0
        assert indicator.first_token_time is None

        indicator.stop()
