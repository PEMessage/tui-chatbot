"""Frontend - UI 渲染逻辑.

职责:
    - 从 main.py 提取渲染逻辑
    - 适配新的 AgentEvent 事件类型
    - 支持工具执行可视化
    - 保持现有的 TPS 统计
"""

from __future__ import annotations

import asyncio
from typing import List, Optional, TYPE_CHECKING, Dict, Any

from .core.events import AgentEvent, AgentEventType
from .indicator import StreamingIndicator

if TYPE_CHECKING:
    from .daemon import Daemon
    from .core.abort_controller import AbortSignal


# 颜色代码
class C:
    """ANSI 颜色代码."""

    GRAY = "\x1b[90m"
    GREEN = "\x1b[92m"
    YELLOW = "\x1b[93m"
    CYAN = "\x1b[96m"
    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"


class Logger:
    """全局调试日志."""

    _instance: Optional["Logger"] = None
    enabled: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def log(self, msg: str) -> None:
        if self.enabled:
            print(f"{C.DIM}[DBG] {msg}{C.RESET}", flush=True)


def log(msg: str) -> None:
    """打印调试日志."""
    Logger().log(msg)


class Frontend:
    """Frontend - 处理 UI 渲染.

    将 AgentEvent 转换为 UI 输出:
        - 推理内容: 灰色
        - 内容: 默认颜色
        - 工具: 黄色标签
        - 统计: 青色
        - 错误: 红色
    """

    # 命令元信息
    META = {"pass_mode": "raw"}

    def __init__(self, daemon: Daemon, session_manager=None):
        """初始化 Frontend.

        Args:
            daemon: Daemon 实例用于对话
            session_manager: 可选的会话管理器
        """
        self.daemon = daemon
        self._session_manager = session_manager

        # 渲染状态
        self._r_started = False
        self._c_started = False
        self._tool_active = False
        self._prev_type: Optional[AgentEventType] = None

        # 统计
        self._stats: Dict[str, Any] = {}

        # 流式指示器
        self._indicator: Optional[StreamingIndicator] = None

    def _reset_state(self) -> None:
        """重置渲染状态."""
        self._r_started = False
        self._c_started = False
        self._tool_active = False
        self._prev_type = None
        self._stats = {}

        # 确保清理遗留的指示器
        if self._indicator:
            self._indicator.stop()
            self._indicator = None

    def _reset_render_state(self) -> None:
        """重置渲染状态 (用于消息生命周期)."""
        self._r_started = False
        self._c_started = False
        self._tool_active = False
        self._prev_type = None
        self._stats = {}

    def _print_header(self, event: AgentEvent) -> None:
        """打印消息头部."""
        # 当前实现不需要额外头部打印
        pass

    def _print_footer(self, event: AgentEvent) -> None:
        """打印消息尾部统计."""
        stats = getattr(event, "stats", None)
        if stats:
            tokens = getattr(stats, "tokens", 0)
            tps = getattr(stats, "tps", 0.0)
            ttft = getattr(stats, "ttft", 0.0)
            print(f"\n[{tokens} tokens | {tps:.1f} TPS | {ttft:.2f}s TTFT]")

    def _print_error(self, event: AgentEvent) -> None:
        """打印错误信息."""
        error = getattr(event, "error", "Unknown error")
        print(f"\n{C.GRAY}[Error: {error}]{C.RESET}")

    def _finalize_output(self) -> None:
        """完成输出并显示统计."""
        # 输出换行
        print()
        # 如果有统计信息，显示格式化后的统计
        if self._stats:
            print(f"{C.CYAN}{self._format_stats(self._stats)}{C.RESET}")

    def _need_separator(self, event_type: AgentEventType) -> bool:
        """检查事件类型切换是否需要分隔符."""
        output_events = {
            AgentEventType.REASONING_TOKEN,
            AgentEventType.CONTENT_TOKEN,
            AgentEventType.MESSAGE_UPDATE,
            AgentEventType.TOOL_EXECUTION_START,
            AgentEventType.TOOL_EXECUTION_END,
            AgentEventType.STATS,
        }

        if self._prev_type is not None and event_type != self._prev_type:
            if event_type in output_events:
                return True
        return False

    def render_event(self, event: AgentEvent) -> None:
        """渲染单个事件.

        Args:
            event: AgentEvent 事件
        """
        # 类型切换时添加分隔符
        if self._need_separator(event.type):
            print("\n")

        self._prev_type = event.type

        # 处理不同类型事件
        match event.type:
            # 生命周期事件
            case AgentEventType.MESSAGE_START:
                # 启动指示器
                if self._indicator:
                    self._indicator.stop()
                self._indicator = StreamingIndicator()
                self._indicator.start()
                self._reset_render_state()

            # Token 事件 (向后兼容)
            case AgentEventType.REASONING_TOKEN:
                # 更新指示器
                if self._indicator:
                    self._indicator.on_token()
                self.on_token(event.data, is_reasoning=True)

            case AgentEventType.CONTENT_TOKEN:
                # 更新指示器
                if self._indicator:
                    self._indicator.on_token()
                self.on_token(event.data, is_reasoning=False)

            # 消息更新事件
            case AgentEventType.MESSAGE_UPDATE:
                # 更新指示器
                if self._indicator and hasattr(event, "token"):
                    self._indicator.on_token()
                self._handle_message_update(event)

            # 消息结束
            case AgentEventType.MESSAGE_END:
                # 停止指示器并显示最终统计
                if self._indicator:
                    self._indicator.stop()
                    self._indicator = None
                self._finalize_output()
                # 重置渲染状态以便下一条消息
                self._reset_render_state()

            # 工具执行事件
            case AgentEventType.TOOL_EXECUTION_START:
                self.on_tool_start(event.tool_name or "unknown", event.args)

            case AgentEventType.TOOL_EXECUTION_END:
                self.on_tool_end(
                    event.tool_name or "unknown",
                    event.result,
                    event.is_error,
                )

            # 统计事件
            case AgentEventType.STATS:
                self._handle_stats(event)

            # 错误事件
            case AgentEventType.ERROR:
                # 错误时停止指示器
                if self._indicator:
                    self._indicator.stop()
                    self._indicator = None
                self._handle_error(event)

            # 其他生命周期事件 (静默)
            case (
                AgentEventType.AGENT_START
                | AgentEventType.AGENT_END
                | AgentEventType.TURN_START
                | AgentEventType.TURN_END
            ):
                pass

            # 其他
            case _:
                log(f"Unknown event type: {event.type}")

    def _handle_message_update(self, event: AgentEvent) -> None:
        """处理消息更新事件."""
        partial = getattr(event, "partial_result", None)
        if not partial:
            return

        msg_type = partial.get("type")
        content = partial.get("content", "")

        if msg_type == "reasoning":
            if not self._r_started:
                print(f"{C.GRAY}[Reasoning]: {C.RESET}", end="", flush=True)
                self._r_started = True
            print(f"{C.GRAY}{content}{C.RESET}", end="", flush=True)

        elif msg_type == "content":
            if not self._c_started:
                print(f"[Assistant]: ", end="", flush=True)
                self._c_started = True
            print(content, end="", flush=True)

    def _handle_stats(self, event: AgentEvent) -> None:
        """处理统计事件."""
        stats = getattr(event, "stats", None)
        if stats:
            self._stats = stats
            print(f"\n{C.CYAN}{self._format_stats(stats)}{C.RESET}")

    def _handle_error(self, event: AgentEvent) -> None:
        """处理错误事件."""
        error = getattr(event, "error", "Unknown error")
        print(f"\n{C.GRAY}[Error: {error}]{C.RESET}")

    def _format_stats(self, stats: Dict[str, Any]) -> str:
        """格式化统计信息."""
        tokens = stats.get("tokens", 0)
        r_tokens = stats.get("r_tokens", 0)
        c_tokens = stats.get("c_tokens", 0)
        elapsed = stats.get("elapsed", 0.0)
        tps = stats.get("tps", 0.0)
        ttft = stats.get("ttft", 0.0)

        if tokens == 0:
            return "[0 TOKENS | 0.0s | TPS 0.0]"

        r_pct = (r_tokens / tokens) * 100 if tokens > 0 else 0
        c_pct = (c_tokens / tokens) * 100 if tokens > 0 else 0

        return (
            f"[{tokens} TOKENS | {r_pct:.1f}% REASONING | {c_pct:.1f}% CONTENT | "
            f"{elapsed:.1f}s | TPS {tps:.1f} | TTFT {ttft:.2f}s]"
        )

    def on_token(self, token: str, is_reasoning: bool = False) -> None:
        """处理 Token 输出 (向后兼容).

        Args:
            token: Token 文本
            is_reasoning: 是否为推理 token
        """
        if is_reasoning:
            if not self._r_started:
                print(f"{C.GRAY}[Reasoning]: {C.RESET}", end="", flush=True)
                self._r_started = True
            print(f"{C.GRAY}{token}{C.RESET}", end="", flush=True)
        else:
            if not self._c_started:
                print(f"[Assistant]: ", end="", flush=True)
                self._c_started = True
            print(token, end="", flush=True)

    def on_tool_start(self, tool_name: str, args: Optional[Dict] = None) -> None:
        """处理工具开始事件.

        Args:
            tool_name: 工具名称
            args: 工具参数
        """
        self._tool_active = True
        args_str = str(args) if args else ""
        if len(args_str) > 50:
            args_str = args_str[:50] + "..."
        print(f"\n{C.YELLOW}[Tool: {tool_name}]{C.RESET}", end="", flush=True)
        if args_str:
            print(f" {C.DIM}{args_str}{C.RESET}", end="", flush=True)
        print(" → ", end="", flush=True)

    def on_tool_end(self, tool_name: str, result: Any, is_error: bool = False) -> None:
        """处理工具结束事件.

        Args:
            tool_name: 工具名称
            result: 工具执行结果
            is_error: 是否出错
        """
        self._tool_active = False
        color = C.GRAY if is_error else C.GREEN
        result_str = str(result) if result else "Done"
        if len(result_str) > 100:
            result_str = result_str[:100] + "..."
        print(f"{color}{result_str}{C.RESET}")

    async def run(self, argv: List[str]) -> None:
        """运行对话命令.

        Args:
            argv: 命令参数，argv[0] 为完整输入文本
        """
        text = argv[0] if argv else ""
        if not text.strip():
            return

        self._reset_state()

        try:
            stream = self.daemon.chat(text)
            async for event in stream:
                self.render_event(event)

            # 确保最后有换行
            print()

            # 同步 Daemon 消息到 SessionManager
            self._sync_messages_to_session()

        except asyncio.CancelledError:
            print(f"\n{C.GRAY}[Stop]{C.RESET}")
            raise

    def _sync_messages_to_session(self) -> None:
        """将 Daemon 的消息同步到当前会话."""
        if not self._session_manager or not self._session_manager.current():
            return

        from ..agent.types import UserMessage, AssistantMessage, TextContent

        session = self._session_manager.current()
        daemon_messages = self.daemon.messages

        # 获取会话中已有的消息数量，避免重复添加
        existing_count = len(session.messages)

        # 从现有消息之后开始同步新消息
        for msg in daemon_messages[existing_count:]:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user" and content:
                session.add_message(UserMessage(content=content))
            elif role == "assistant" and content:
                session.add_message(
                    AssistantMessage(content=[TextContent(text=content)])
                )
            # 忽略 system 消息


class ModelCommand:
    """/model - 列出或切换模型."""

    META = {"pass_mode": "shlex"}

    def __init__(self, daemon: Daemon):
        self.daemon = daemon

    async def run(self, argv: List[str]) -> None:
        """运行模型命令."""
        if len(argv) > 1:
            # /model <name> - 切换模型
            self.daemon.switch_model(argv[1])
            print(f"Switched: {argv[1]}")
        else:
            # /model - 列出模型
            print("Fetching...")
            models = await self.daemon.list_models()
            if models:
                current = self.daemon.model
                print(f"\nModels (current: {current}):")
                print("-" * 40)
                for m in sorted(models):
                    marker = " *" if m == current else ""
                    print(f"  {m}{marker}")
                print("-" * 40)
            else:
                print("No models available")


class ClearCommand:
    """/clear - 清除历史."""

    META = {"pass_mode": "shlex"}

    def __init__(self, daemon: Daemon):
        self.daemon = daemon

    async def run(self, argv: List[str]) -> None:
        """运行清除命令."""
        self.daemon.clear_history()
        print("Cleared")


class HelpCommand:
    """/help - 显示帮助."""

    META = {"pass_mode": "shlex"}

    async def run(self, argv: List[str]) -> None:
        """运行帮助命令."""
        print("""
Commands:
  <text>                    Chat with bot (default)
  /model [name]             List or switch models
  /search <keyword>         Search messages in session history
                            Options: --regex, --user-only, --assistant-only
  /export [--json] [--all]  Export session(s) to file
                            --json: Export as JSON (default: Markdown)
                            --all: Export all sessions (default: current)
  /save                     Save current session immediately
  /sessions                 List all sessions
  /load <session_id>        Load a specific session
  /config [key] [value]     View or modify configuration
  /clear                    Clear history
  /help                     This help
  /quit, /exit              Exit

Press Ctrl-C to interrupt chat.
        """)


class QuitException(Exception):
    """退出异常，用于通知 Shell 退出循环."""

    pass


class QuitCommand:
    """/quit, /exit - 退出."""

    META = {"pass_mode": "shlex"}

    async def run(self, argv: List[str]) -> None:
        """运行退出命令."""
        print("Bye!")
        raise QuitException()


class SearchCommand:
    """搜索命令 /search <keyword> [--regex] [--user-only] [--assistant-only]"""

    META = {
        "pass_mode": "shlex",
        "name": "search",
        "help": "在会话历史中搜索消息",
        "args": ["keyword"],
        "options": ["--regex", "--user-only", "--assistant-only"],
    }

    def __init__(self, session_manager, frontend):
        self._session_manager = session_manager
        self._frontend = frontend

    async def run(self, argv: List[str]) -> None:
        """运行搜索命令."""
        if len(argv) < 2:
            print("Usage: /search <keyword> [--regex] [--user-only] [--assistant-only]")
            return

        # 解析参数
        args = argv[1:]  # 去掉命令名
        keyword = args[0]
        use_regex = "--regex" in args

        from ..search import MessageSearchEngine, SearchScope

        current = self._session_manager.current()
        if not current:
            print("No active session")
            return

        engine = MessageSearchEngine()
        engine.index_session(current)

        scope = SearchScope.ALL
        if "--user-only" in args:
            scope = SearchScope.USER_ONLY
        elif "--assistant-only" in args:
            scope = SearchScope.ASSISTANT_ONLY

        result = engine.search(keyword, scope=scope, use_regex=use_regex)

        if not result.matches:
            print(f"No messages found matching '{keyword}'")
            return

        print(f"\nFound {result.total_matches} results:\n")
        for i, match in enumerate(result.matches[:10], 1):
            role = getattr(match.message, "role", "unknown")
            # 高亮匹配内容
            highlighted = f"{match.context_before}\033[1m{match.matched_text}\033[0m{match.context_after}"
            print(f"{i}. [{role}] ...{highlighted}...")


class ExportCommand:
    """导出命令 /export [--json] [--all] [session_id]"""

    META = {
        "pass_mode": "shlex",
        "name": "export",
        "help": "导出会话到文件",
        "args": ["[session_id]"],
        "options": ["--json", "--all"],
    }

    def __init__(self, session_manager):
        self._session_manager = session_manager

    async def run(self, argv: List[str]) -> None:
        """运行导出命令."""
        from ..export import SessionExporter, ExportFormat

        args = argv[1:]  # 去掉命令名
        use_json = "--json" in args
        format_type = ExportFormat.JSON if use_json else ExportFormat.MARKDOWN

        if "--all" in args:
            sessions = self._session_manager.list_all()
            if not sessions:
                print("No sessions to export")
                return
            exporter = SessionExporter()
            paths = exporter.export_batch(sessions, format_type)
            print(f"Exported {len(paths)} sessions")
        else:
            current = self._session_manager.current()
            if not current:
                print("No active session")
                return
            exporter = SessionExporter()
            path = exporter.export(current, format_type)
            print(f"Exported to: {path}")


class SaveCommand:
    """保存命令 /save"""

    META = {
        "pass_mode": "shlex",
        "name": "save",
        "help": "立即保存当前会话",
        "args": [],
    }

    def __init__(self, session_manager):
        self._session_manager = session_manager

    async def run(self, argv: List[str]) -> None:
        """运行保存命令."""
        current = self._session_manager.current()
        if not current:
            print("No active session to save")
            return
        success = self._session_manager.save_current()
        if success:
            print(f"Session '{current.metadata.title}' saved")
        else:
            print("Failed to save session")


class SessionsCommand:
    """列出所有会话 /sessions"""

    META = {
        "pass_mode": "shlex",
        "name": "sessions",
        "help": "列出所有会话",
        "args": [],
    }

    def __init__(self, session_manager):
        self._session_manager = session_manager

    async def run(self, argv: List[str]) -> None:
        """运行列出会话命令."""
        sessions = self._session_manager.list_all()
        current = self._session_manager.current()
        current_id = current.metadata.id if current else None

        if not sessions:
            print("No sessions found")
            return

        print("\nSessions:")
        for s in sessions:
            marker = " *" if s.metadata.id == current_id else ""
            msg_count = len(s.messages) if s.messages else 0
            print(f"  {s.metadata.id}{marker}: {s.metadata.title} ({msg_count} msgs)")


class LoadCommand:
    """加载会话 /load <session_id>"""

    META = {
        "pass_mode": "shlex",
        "name": "load",
        "help": "加载指定会话",
        "args": ["session_id"],
    }

    def __init__(self, session_manager):
        self._session_manager = session_manager

    async def run(self, argv: List[str]) -> None:
        """运行加载命令."""
        if len(argv) < 2:
            print("Usage: /load <session_id>")
            return
        session_id = argv[1]
        session = self._session_manager.load(session_id)
        if session:
            print(f"Loaded session: {session.metadata.title}")
        else:
            print(f"Session {session_id} not found")


class ConfigCommand:
    """配置命令 /config [key] [value]"""

    META = {
        "pass_mode": "shlex",
        "name": "config",
        "help": "查看或修改配置",
        "args": ["[key]", "[value]"],
        "examples": [
            "/config - 显示所有配置",
            "/config temperature - 查看温度设置",
            "/config temperature 0.5 - 修改温度",
        ],
    }

    # 可配置项及其说明
    CONFIG_ITEMS = {
        "default_model": ("默认模型名称", str),
        "default_provider": ("默认 Provider", str),
        "temperature": ("采样温度 (0.0-2.0)", float),
        "max_tokens": ("最大 token 数", int),
        "theme": ("主题名称", str),
        "auto_save": ("自动保存配置", bool),
    }

    def __init__(self, config_manager):
        self._config_manager = config_manager

    async def run(self, argv: List[str]) -> None:
        """运行配置命令."""
        args = argv[1:]  # 去掉命令名

        if not args:
            self._show_all_config()
            return

        key = args[0]

        if key not in self.CONFIG_ITEMS:
            print(f"Unknown config key: {key}")
            print(f"\nAvailable keys:")
            for k, (desc, _) in self.CONFIG_ITEMS.items():
                print(f"  {k:<20} - {desc}")
            return

        if len(args) < 2:
            # 查看单项配置
            self._show_config_item(key)
        else:
            # 修改配置
            self._set_config(key, args[1])

    def _show_all_config(self) -> None:
        """显示所有配置."""
        config = self._config_manager.get()

        print(f"\n{C.BOLD}Current Configuration:{C.RESET}\n")

        for key, (desc, _) in self.CONFIG_ITEMS.items():
            value = getattr(config, key)
            print(f"  {C.CYAN}{key:<20}{C.RESET} {value}")
            print(f"  {'':20} {C.DIM}{desc}{C.RESET}")
            print()

        config_file = self._config_manager._config_file
        print(f"{C.DIM}Config file: {config_file}{C.RESET}")

    def _show_config_item(self, key: str) -> None:
        """显示单项配置."""
        config = self._config_manager.get()
        value = getattr(config, key)
        desc, type_ = self.CONFIG_ITEMS[key]

        print(f"\n{C.BOLD}{key}{C.RESET}")
        print(f"  Value: {C.CYAN}{value}{C.RESET}")
        print(f"  Type:  {type_.__name__}")
        print(f"  Desc:  {desc}")

    def _set_config(self, key: str, value_str: str) -> None:
        """设置配置值."""
        _, type_ = self.CONFIG_ITEMS[key]

        try:
            # 类型转换
            if type_ == bool:
                value = value_str.lower() in ("true", "1", "yes", "on")
            elif type_ == int:
                value = int(value_str)
            elif type_ == float:
                value = float(value_str)
            else:
                value = value_str

            self._config_manager.update(**{key: value})
            print(f"✓ {key} = {value}")

        except ValueError as e:
            print(f"✗ Invalid value: {e}")
            print(f"  Expected type: {type_.__name__}")
