"""TUI Chatbot - 更新后的入口点.

使用新架构:
    - 新模块: daemon, frontend, provider, agent
    - ProviderRegistry 管理 LLM 提供商
    - ToolRegistry 管理工具
    - AgentEvent 事件系统

保持向后兼容:
    - CLI 参数 (--api-key, --base-url, --model, --debug, -c)
    - 环境变量 (OPENAI_API_KEY 等)
    - 交互式命令 (/model, /clear, /help, /quit)
"""

from __future__ import annotations

import os
import sys
import asyncio
import shlex
from typing import List, Optional
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ╭────────────────────────────────────────────────────────────╮
# │  导入新架构模块                                             │
# ╰────────────────────────────────────────────────────────────╯

try:
    # 作为包导入时使用相对导入
    from .daemon import Daemon, LegacyDaemon
    from .frontend import (
        Frontend,
        ModelCommand,
        ClearCommand,
        HelpCommand,
        QuitCommand,
        Logger as FrontendLogger,
        C,
    )
    from .provider.registry import (
        ProviderRegistry,
        register_default_providers,
        create_provider_from_env,
    )
    from .provider.openai_provider import (
        OpenAIProvider,
        OpenAIProviderConfig,
    )
    from .agent.tool import (
        ToolRegistry,
        create_default_tool_registry,
    )
except ImportError:
    # 直接运行时回退到绝对导入
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tui_chatbot.daemon import Daemon, LegacyDaemon
    from tui_chatbot.frontend import (
        Frontend,
        ModelCommand,
        ClearCommand,
        HelpCommand,
        QuitCommand,
        Logger as FrontendLogger,
        C,
    )
    from tui_chatbot.provider.registry import (
        ProviderRegistry,
        register_default_providers,
        create_provider_from_env,
    )
    from tui_chatbot.provider.openai_provider import (
        OpenAIProvider,
        OpenAIProviderConfig,
    )
    from tui_chatbot.agent.tool import (
        ToolRegistry,
        create_default_tool_registry,
    )


# ╭────────────────────────────────────────────────────────────╮
# │  配置 (向后兼容)                                            │
# ╰────────────────────────────────────────────────────────────╯

ReasoningEffort = Literal["minimal", "low", "medium", "high"]


@dataclass(frozen=True)
class Config:
    """向后兼容的配置类."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    debug: bool = False
    history: int = 10
    reasoning_effort: Optional[ReasoningEffort] = None


# ╭────────────────────────────────────────────────────────────╮
# │  初始化                                                     │
# ╰────────────────────────────────────────────────────────────╯


def initialize_providers(cfg: Config) -> bool:
    """初始化 ProviderRegistry.

    Args:
        cfg: 配置对象

    Returns:
        bool: 是否成功初始化至少一个 Provider
    """
    # 1. 首先尝试从环境变量注册默认提供商
    register_default_providers()

    # 2. 如果有 api_key，注册 OpenAI Provider
    if cfg.api_key:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

            # 构建配置
            openai_config = OpenAIProviderConfig(
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                model=cfg.model,
                reasoning_effort=cfg.reasoning_effort,
            )

            provider = OpenAIProvider(client=client, config=openai_config)

            # 注册 Provider
            ProviderRegistry.register("openai", provider)
            ProviderRegistry.register("openai-chat", provider)

            return True

        except ImportError as e:
            print(f"Warning: Failed to import openai: {e}", file=sys.stderr)
            return False
        except Exception as e:
            print(
                f"Warning: Failed to initialize OpenAI provider: {e}", file=sys.stderr
            )
            return False

    return len(ProviderRegistry.list()) > 0


def initialize_tools() -> ToolRegistry:
    """初始化 ToolRegistry 并注册默认工具.

    Returns:
        ToolRegistry: 包含默认工具的注册表
    """
    return create_default_tool_registry()


# ╭────────────────────────────────────────────────────────────╮
# │  Shell (命令路由)                                           │
# ╰────────────────────────────────────────────────────────────╯


class Shell:
    """Shell - 统一命令路由."""

    PROMPT = ">>> "

    def __init__(self, cfg: Config):
        """初始化 Shell.

        Args:
            cfg: 配置对象
        """
        self.cfg = cfg

        # 初始化 Provider
        has_provider = initialize_providers(cfg)

        # 初始化 Tools
        tool_registry = initialize_tools()

        # 创建 Daemon (优先使用新架构，兼容旧方式)
        if has_provider and ProviderRegistry.get("openai"):
            # 使用新架构 Daemon
            self.daemon = Daemon(
                provider_api="openai",
                model=cfg.model,
                history_limit=cfg.history,
                tool_registry=tool_registry,
            )
        else:
            # 回退到旧架构 Daemon
            self.daemon = LegacyDaemon(cfg)

        # 创建命令
        self.cmds = {
            "__default__": Frontend(self.daemon),
            "model": ModelCommand(self.daemon),
            "m": ModelCommand(self.daemon),
            "clear": ClearCommand(self.daemon),
            "c": ClearCommand(self.daemon),
            "help": HelpCommand(),
            "h": HelpCommand(),
            "quit": QuitCommand(),
            "q": QuitCommand(),
            "exit": QuitCommand(),
        }

    def _get_cmd(self, line: str) -> tuple:
        """解析命令.

        Args:
            line: 输入行

        Returns:
            (command, argv) 元组
        """
        is_command = line.startswith("/")
        cmd_name = ""
        cmd = None

        if is_command:
            # 查看命令名
            parts = line.split(None, 1)
            if parts:
                cmd_name = parts[0][1:].lower()
                cmd = self.cmds.get(cmd_name)

        if not is_command or not cmd:
            # 默认：使用 Frontend (聊天)
            cmd = self.cmds["__default__"]
            pass_mode = getattr(cmd, "META", {}).get("pass_mode", "raw")
            if pass_mode == "raw":
                return cmd, [line]
            else:
                return cmd, shlex.split(line)

        # 已知命令 - 检查 META
        pass_mode = getattr(cmd, "META", {}).get("pass_mode", "shlex")

        if pass_mode == "raw":
            if " " in line:
                _, rest = line.split(None, 1)
                return cmd, [rest]
            return cmd, [""]
        else:
            try:
                argv = shlex.split(line)
            except ValueError:
                argv = line.split()
            return cmd, argv

    async def run(self) -> None:
        """主循环."""
        # 显示欢迎信息
        provider = ProviderRegistry.get("openai")
        if provider:
            print(f"🤖 Provider: {provider.name} | Model: {self.daemon.model}")
        else:
            print(f"🤖 Model: {self.daemon.model}")

        print("Commands: /model, /clear, /help, /quit")
        print("Default: chat (type anything)\n")

        if not provider:
            print("⚠️  No API key. Set OPENAI_API_KEY or use --api-key")

        while True:
            try:
                line = input(self.PROMPT).strip()
                if not line:
                    continue

                cmd, argv = self._get_cmd(line)
                task = asyncio.create_task(cmd.run(argv))
                try:
                    await task
                except asyncio.CancelledError:
                    pass  # Ctrl-C 已处理

            except KeyboardInterrupt:
                print(f"\n{C.GRAY}[Interrupt]{C.RESET}")
            except EOFError:
                print("\nBye!")
                break
            except Exception as e:
                print(f"\n[Error: {e}]")
                if FrontendLogger.enabled:
                    import traceback

                    traceback.print_exc()


# ╭────────────────────────────────────────────────────────────╮
# │  入口点                                                     │
# ╰────────────────────────────────────────────────────────────╯


async def run_command(cfg: Config, line: str) -> int:
    """执行单条命令 (非交互模式).

    Args:
        cfg: 配置
        line: 命令行

    Returns:
        int: 退出码 (0=成功, 1=错误)
    """
    shell = Shell(cfg)
    cmd, argv = shell._get_cmd(line)

    try:
        await cmd.run(argv)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> None:
    """主入口点."""
    import argparse

    parser = argparse.ArgumentParser(
        description="TUI Chatbot - New Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s                           # Interactive mode
  %(prog)s --model o3-mini --reason-effort high
  %(prog)s -c "hello world"          # Single chat message
  %(prog)s -c "/model gpt-4"         # Switch model
        """,
    )

    # CLI 参数 (与环境变量集成)
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="API base URL (env: OPENAI_BASE_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="API key (env: OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        help="Model name (env: OPENAI_MODEL)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--reason-effort",
        choices=["minimal", "low", "medium", "high"],
        default=os.getenv("REASONING_EFFORT"),
        help="Reasoning effort for reasoning models (env: REASONING_EFFORT)",
    )
    parser.add_argument(
        "-c",
        dest="command",
        help="Execute a single command and exit",
    )

    args = parser.parse_args()

    # 启用调试日志
    FrontendLogger.enabled = args.debug

    # 创建配置
    cfg = Config(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        debug=args.debug,
        reasoning_effort=args.reason_effort,
    )

    # 非交互模式
    if args.command:
        exit_code = asyncio.run(run_command(cfg, args.command))
        sys.exit(exit_code)

    # 交互模式
    shell = Shell(cfg)
    try:
        asyncio.run(shell.run())
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()
