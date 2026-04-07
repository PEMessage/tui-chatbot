"""
Minimal TUI Chatbot - Daemon Architecture

Design:
    - ChatBotDaemon: Stateful service (context memory)
    - Frontend: One-shot command (like `ls`, `cat`)
    - Commands: Also one-shot, but talk to daemon
    - All operations are interruptible (Ctrl-C)
    - Unified interface: all return AsyncGenerator[Event, None]

Analogy:
    - ChatBotDaemon = PostgreSQL server
    - Frontend = psql client (default connection)
    - Commands = sql commands via psql
"""

import os
import sys
import time
import argparse
import asyncio
from typing import List, Dict, Optional, Any, NamedTuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum, auto

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


# ╭────────────────────────────────────────────────────────────╮
# │  Global Logger                                             │
# ╰────────────────────────────────────────────────────────────╯


class Logger:
    """Global singleton logger."""

    _instance: Optional["Logger"] = None
    enabled: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def log(self, msg: str) -> None:
        if self.enabled:
            print(f"[DBG] {msg}", flush=True)


def log(msg: str) -> None:
    Logger().log(msg)


# ╭────────────────────────────────────────────────────────────╮
# │  Protocols & Constants                                     │
# ╰────────────────────────────────────────────────────────────╯


class C:
    """ANSI colors."""

    GRAY = "\x1b[90m"
    CYAN = "\x1b[36m"
    RESET = "\x1b[0m"


PROMPT = ">>> "
SYSTEM_MSG = {"role": "system", "content": "You are a helpful assistant."}


class EventType(Enum):
    """All possible event types - unified protocol."""

    OUTPUT = auto()  # Normal output
    STATS = auto()  # Statistics
    ERROR = auto()  # Error message
    DONE = auto()  # Operation complete
    STOP = auto()  # Interrupted


@dataclass(frozen=True)
class Event:
    """Event - the universal interface."""

    type: EventType
    data: Any = None


# ╭────────────────────────────────────────────────────────────╮
# │  Stats                                                     │
# ╰────────────────────────────────────────────────────────────╯


@dataclass
class Stats:
    """Statistics for any operation."""

    tokens: int = 0
    start: float = field(default_factory=time.time)
    first_token: Optional[float] = None

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    @property
    def tps(self) -> float:
        return self.tokens / self.elapsed if self.elapsed > 0 else 0.0

    def on_token(self) -> None:
        if self.first_token is None:
            self.first_token = time.time()

    def __str__(self) -> str:
        if self.tokens == 0:
            return "[0 | 0.0s]\n[TPS 0.0 | TTFT 0.0s]"
        ttft = self.first_token - self.start if self.first_token else 0.0
        return (
            f"[{self.tokens} | {self.elapsed:.1f}s]\n"
            f"[TPS {self.tps:.1f} | TTFT {ttft:.2f}s]"
        )


# ╭────────────────────────────────────────────────────────────╮
# │  ChatBotDaemon (Stateful Service)                          │
# ╰────────────────────────────────────────────────────────────╯


class ChatBotDaemon:
    """Stateful daemon - keeps conversation history.

    Like a database server - maintains state between queries.
    """

    def __init__(self, cfg: "Config"):
        self.cfg = cfg
        self.client: Optional[AsyncOpenAI] = None
        self.msgs: List[Dict[str, str]] = [SYSTEM_MSG]
        self.model = cfg.model

        if cfg.api_key:
            self.client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    # ── Public API (synchronous interface) ─────────────────────────

    def trim_history(self) -> None:
        """Trim messages to max history."""
        max_len = self.cfg.history + 1
        if len(self.msgs) > max_len:
            self.msgs = [self.msgs[0]] + self.msgs[-self.cfg.history :]

    def get_models(self) -> List[str]:
        """Get available models (async operation)."""
        if not self.client:
            return []
        # Returns coroutine - caller must await
        return self._fetch_models()

    def switch_model(self, name: str) -> None:
        """Switch current model."""
        self.model = name

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.msgs = [SYSTEM_MSG]

    # ── Async Operations (generators) ───────────────────────────

    async def _fetch_models(self) -> List[str]:
        """Internal: fetch models from API."""
        if not self.client:
            return []
        try:
            models = await self.client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            return []

    def chat(self, text: str) -> AsyncGenerator[Event, None]:
        """Chat operation - yields events.

        This is the MAIN operation. Like `psql -c "SELECT ..."`
        """
        return self._chat_stream(text)

    async def _chat_stream(self, text: str) -> AsyncGenerator[Event, None]:
        """Internal: streaming chat implementation."""
        if not self.client:
            yield Event(EventType.ERROR, "No API key")
            return

        self.trim_history()
        self.msgs.append({"role": "user", "content": text})

        stats = Stats()
        r_buf, c_buf = "", ""
        r_started, c_started = False, False

        try:
            log(f"chat: model={self.model}")
            stream = await self.client.chat.completions.create(
                model=self.model, messages=self.msgs, stream=True
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                r = getattr(delta, "reasoning_content", None)
                c = getattr(delta, "content", None)

                # Reasoning tokens
                if r:
                    if not r_started:
                        stats.on_token()
                        r_started = True
                        yield Event(EventType.OUTPUT, ("[Reasoning]", C.GRAY))
                    r_buf += r
                    stats.tokens += len(r) // 4
                    yield Event(EventType.OUTPUT, (r, C.GRAY))

                # Content tokens
                if c:
                    if not c_started:
                        if not stats.first_token:
                            stats.on_token()
                        c_started = True
                        if r_started:
                            yield Event(EventType.OUTPUT, ("\n", ""))
                        yield Event(EventType.OUTPUT, ("[Assistant]", ""))
                        yield Event(EventType.OUTPUT, ("Assistant: ", ""))
                    c_buf += c
                    stats.tokens += len(c) // 4
                    yield Event(EventType.OUTPUT, (c, ""))

                log(f"chunk: r={bool(r)} c={bool(c)}")

            # Complete
            resp = c_buf or r_buf
            if resp:
                self.msgs.append({"role": "assistant", "content": resp})
                yield Event(EventType.OUTPUT, ("\n", ""))
                yield Event(EventType.STATS, stats)
                yield Event(EventType.DONE, None)
            else:
                yield Event(EventType.ERROR, "Empty response")

        except asyncio.CancelledError:
            # Interrupted - save partial
            resp = c_buf or r_buf
            if resp:
                self.msgs.append({"role": "assistant", "content": resp})
                yield Event(EventType.OUTPUT, ("\n", ""))
                yield Event(EventType.STOP, stats)
            raise


# ╭────────────────────────────────────────────────────────────╮
# │  Config                                                    │
# ╰────────────────────────────────────────────────────────────╯


@dataclass(frozen=True)
class Config:
    """Bot configuration."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    debug: bool = False
    history: int = 10


# ╭────────────────────────────────────────────────────────────╮
# │  Operation (Unified Interface)                               │
# ╰────────────────────────────────────────────────────────────╯


class Operation:
    """Any operation that can be executed and interrupted.

    Both Frontend (chat) and Commands use this interface.
    """

    def __init__(self, daemon: ChatBotDaemon):
        self.daemon = daemon

    async def run(self, *args, **kwargs) -> AsyncGenerator[Event, None]:
        """Execute operation - must be implemented by subclasses."""
        raise NotImplementedError


class ChatOperation(Operation):
    """Chat operation - the default operation."""

    async def run(self, text: str) -> AsyncGenerator[Event, None]:
        """Run chat - delegates to daemon."""
        async for ev in self.daemon.chat(text):
            yield ev


class ListModelsOperation(Operation):
    """List models operation."""

    async def run(self) -> AsyncGenerator[Event, None]:
        """Fetch and yield models."""
        if not self.daemon.client:
            yield Event(EventType.ERROR, "No API key")
            return

        stats = Stats()
        yield Event(EventType.OUTPUT, ("Fetching...", ""))

        try:
            models = await self.daemon._fetch_models()
            stats.on_token()
            stats.tokens = len(models)

            if models:
                lines = [f"Models (current: {self.daemon.model}):"]
                lines.append("-" * 40)
                for m in sorted(models):
                    marker = " *" if m == self.daemon.model else ""
                    lines.append(f"  {m}{marker}")
                lines.append("-" * 40)
                yield Event(EventType.OUTPUT, ("\n".join(lines), ""))
                yield Event(EventType.STATS, stats)
                yield Event(EventType.DONE, None)
            else:
                yield Event(EventType.ERROR, "No models available")

        except asyncio.CancelledError:
            yield Event(EventType.STOP, stats)
            raise


class SwitchModelOperation(Operation):
    """Switch model operation."""

    async def run(self, name: str) -> AsyncGenerator[Event, None]:
        """Switch model."""
        self.daemon.switch_model(name)
        yield Event(EventType.OUTPUT, (f"Switched: {name}", ""))
        yield Event(EventType.DONE, None)


class ClearOperation(Operation):
    """Clear history operation."""

    async def run(self) -> AsyncGenerator[Event, None]:
        """Clear conversation."""
        self.daemon.clear_history()
        yield Event(EventType.OUTPUT, ("Cleared", ""))
        yield Event(EventType.DONE, None)


class HelpOperation(Operation):
    """Help operation."""

    async def run(self) -> AsyncGenerator[Event, None]:
        """Show help."""
        help_text = """
Commands:
  <text>          Chat with bot (default)
  /model          List models
  /model <name>   Switch model
  /clear          Clear history
  /help           This help
  /quit, /exit    Exit

Press Ctrl-C to interrupt any operation.
        """.strip()
        yield Event(EventType.OUTPUT, (help_text, ""))
        yield Event(EventType.DONE, None)


# ╭────────────────────────────────────────────────────────────╮
# │  Renderer (UI Layer)                                       │
# ╰────────────────────────────────────────────────────────────╯


class Renderer:
    """Render events to terminal."""

    @staticmethod
    def render(ev: Event) -> None:
        """Render single event."""
        match ev.type:
            case EventType.OUTPUT:
                text, color = ev.data
                if color:
                    print(f"{color}{text}{C.RESET}", end="", flush=True)
                else:
                    print(text, end="", flush=True)
            case EventType.STATS:
                print(f"{ev.data}")
            case EventType.ERROR:
                print(f"\n[Error: {ev.data}]")
            case EventType.DONE:
                pass  # Silent
            case EventType.STOP:
                print(f"\n{C.GRAY}[Stop]{C.RESET}")
                print(f"{C.GRAY}{ev.data}{C.RESET}")


# ╭────────────────────────────────────────────────────────────╮
# │  Shell (Router)                                            │
# ╰────────────────────────────────────────────────────────────╯


class Shell:
    """Shell - routes input to operations.

    Like bash: parses input, creates operation, executes.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Single daemon instance (like PostgreSQL server)
        self.daemon = ChatBotDaemon(cfg)
        self.renderer = Renderer()

    def _parse(self, line: str) -> tuple[Operation, tuple, dict]:
        """Parse input into (operation, args, kwargs).

        Returns ChatOperation for non-command text.
        """
        if not line.startswith("/"):
            # DEFAULT: Chat (like default command in shell)
            return ChatOperation(self.daemon), (line,), {}

        parts = line.split(maxsplit=1)
        cmd = parts[0][1:].lower()
        args = parts[1] if len(parts) > 1 else ""

        match cmd:
            case "model" | "m":
                if args:
                    return SwitchModelOperation(self.daemon), (args,), {}
                return ListModelsOperation(self.daemon), (), {}
            case "clear" | "c":
                return ClearOperation(self.daemon), (), {}
            case "help" | "h":
                return HelpOperation(self.daemon), (), {}
            case "quit" | "q" | "exit":
                print("Bye!")
                sys.exit(0)
            case _:
                # Unknown command - treat as chat
                return ChatOperation(self.daemon), (line,), {}

    async def _execute(self, op: Operation, args: tuple, kwargs: dict) -> None:
        """Execute operation with cancellation support."""
        task = asyncio.create_task(self._consume(op, args, kwargs))
        try:
            await task
        except asyncio.CancelledError:
            # Operation was interrupted
            pass

    async def _consume(self, op: Operation, args: tuple, kwargs: dict) -> None:
        """Consume operation events."""
        try:
            async for ev in op.run(*args, **kwargs):
                self.renderer.render(ev)
        except asyncio.CancelledError:
            raise

    async def run(self) -> None:
        """Main loop."""
        print(f"🤖 {self.cfg.base_url} | {self.daemon.model}")
        print("Commands: /model, /clear, /help, /quit")
        print("Default: chat (type anything)\n")

        if not self.daemon.client:
            print("⚠️  No API key set")

        while True:
            try:
                line = input(PROMPT).strip()
                if not line:
                    continue

                # Parse into operation
                op, args, kwargs = self._parse(line)

                # Execute with Ctrl-C support
                await self._execute(op, args, kwargs)

            except KeyboardInterrupt:
                print(f"\n{C.GRAY}[Interrupt]{C.RESET}")
            except EOFError:
                print("\nBye!")
                break
            except Exception as e:
                print(f"\n[Error: {e}]")
                log(f"error: {type(e).__name__}")


# ╭────────────────────────────────────────────────────────────╮
# │  Entry                                                     │
# ╰────────────────────────────────────────────────────────────╯


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal TUI Chatbot - Daemon Architecture",
        epilog="""
Examples:
  %(prog)s                    # Interactive shell
  %(prog)s --api-key sk-xxx   # With custom key
        """,
    )

    parser.add_argument(
        "--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    Logger.enabled = args.debug

    cfg = Config(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        debug=args.debug,
    )

    shell = Shell(cfg)

    try:
        asyncio.run(shell.run())
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()
