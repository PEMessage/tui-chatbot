"""加载指示器 - 流式响应视觉反馈."""

from __future__ import annotations

import sys
import time
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class StreamingIndicator:
    """流式响应指示器.

    显示实时统计信息:
        - TTFT (首 token 延迟)
        - Token 计数
        - 生成速度

    Usage:
        >>> indicator = StreamingIndicator()
        >>> indicator.start()
        >>> # 流式生成中...
        >>> indicator.update(tokens=10)
        >>> indicator.stop()
    """

    start_time: float = field(default_factory=time.time)
    first_token_time: Optional[float] = None
    token_count: int = 0
    _active: bool = False

    # 颜色代码
    GRAY = "\x1b[90m"
    CYAN = "\x1b[96m"
    GREEN = "\x1b[92m"
    YELLOW = "\x1b[93m"
    RESET = "\x1b[0m"
    CLEAR_LINE = "\r\033[K"

    def start(self) -> None:
        """开始指示器."""
        self.start_time = time.time()
        self.first_token_time = None
        self.token_count = 0
        self._active = True
        self._display()

    def on_first_token(self) -> None:
        """记录首 token 时间."""
        if self.first_token_time is None:
            self.first_token_time = time.time()

    def on_token(self) -> None:
        """收到 token."""
        if self.first_token_time is None:
            self.on_first_token()
        self.token_count += 1
        if self._active:
            self._display()

    def update(self, tokens: int = 1) -> None:
        """更新 token 计数.

        Args:
            tokens: 新增 token 数
        """
        if not self._active:
            return

        if self.first_token_time is None:
            self.on_first_token()

        self.token_count += tokens
        self._display()

    def stop(self) -> None:
        """停止指示器并清除显示."""
        self._active = False
        sys.stdout.write(self.CLEAR_LINE)
        sys.stdout.flush()

    @property
    def ttft(self) -> float:
        """首次 token 延迟 (Time To First Token)."""
        if self.first_token_time:
            return self.first_token_time - self.start_time
        return time.time() - self.start_time

    @property
    def tps(self) -> float:
        """每秒 token 数 (Tokens Per Second)."""
        if self.first_token_time:
            elapsed = time.time() - self.first_token_time
            if elapsed > 0 and self.token_count > 0:
                return self.token_count / elapsed
        return 0.0

    def render(self) -> str:
        """渲染状态 (简化版字符串，用于非终端显示)."""
        if self.first_token_time is None:
            return f"🤔 Thinking... ({self.ttft:.1f}s)"
        else:
            return f"✍️  Generating... ({self.token_count} tokens, {self.tps:.1f} tps)"

    def _display(self) -> None:
        """显示当前状态."""
        # 计算 TTFT
        if self.first_token_time:
            ttft = self.first_token_time - self.start_time
            ttft_str = f"{ttft:.2f}s"
            status = f"{self.GREEN}Generating{self.RESET}"
        else:
            ttft_str = f"{self.ttft:.1f}s"
            status = f"{self.CYAN}Thinking{self.RESET}"

        # 计算 TPS
        tps_str = f"{self.tps:.1f}"

        # 格式化显示
        indicator = (
            f"{self.CLEAR_LINE}"
            f"{self.GRAY}[{status} | "
            f"TTFT {ttft_str} | "
            f"{self.token_count} tokens | "
            f"TPS {tps_str}]{self.RESET}"
        )

        sys.stdout.write(indicator)
        sys.stdout.flush()
