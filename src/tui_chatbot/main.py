"""
Minimal TUI Chatbot - Shell Architecture

Design:
    - Shell: Like bash, routes commands
    - Daemon: Stateful context (msgs, model)
    - Commands: All callable, including Frontend (chat)
    - Frontend is just a Command that consumes daemon's generator

Analogy:
    - Daemon = PostgreSQL server (stateful)
    - Commands = SQL operations
    - Frontend = "SELECT ..." (complex streaming query)
    - /model = "\dt" (simple meta query)
"""

import os
import sys
import time
import asyncio
from typing import List, Dict, Optional, Any, NamedTuple, AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


# ╭────────────────────────────────────────────────────────────╮
# │  Logger                                                    │
# ╰────────────────────────────────────────────────────────────╯


class Logger:
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
# │  Constants                                                 │
# ╰────────────────────────────────────────────────────────────╯


class C:
    GRAY = "\x1b[90m"
    CYAN = "\x1b[36m"
    RESET = "\x1b[0m"


PROMPT = ">>> "
SYSTEM_MSG = {"role": "system", "content": "You are a helpful assistant."}


class EventType(Enum):
    OUTPUT = auto()
    STATS = auto()
    ERROR = auto()
    DONE = auto()


@dataclass(frozen=True)
class Event:
    type: EventType
    data: Any = None


# ╭────────────────────────────────────────────────────────────╮
# │  Stats                                                     │
# ╰────────────────────────────────────────────────────────────╯


@dataclass
class Stats:
    tokens: int = 0
    r_tokens: int = 0
    c_tokens: int = 0
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
            return "[0 | 0% | 0% | 0.0s]\n[TPS 0.0 | AVG 0.0 | TTFT 0.0s]"
        r_pct = (self.r_tokens / self.tokens) * 100
        c_pct = (self.c_tokens / self.tokens) * 100
        ttft = self.first_token - self.start if self.first_token else 0.0
        return (
            f"[{self.tokens} | {r_pct:.1f}% | {c_pct:.1f}% | {self.elapsed:.1f}s]\n"
            f"[TPS {self.tps:.1f} | AVG {self.tps:.1f} | TTFT {ttft:.2f}s]"
        )


# ╭────────────────────────────────────────────────────────────╮
# │  Daemon (Stateful Service)                                 │
# ╰────────────────────────────────────────────────────────────╯


@dataclass(frozen=True)
class Config:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    debug: bool = False
    history: int = 10


class Daemon:
    """Stateful daemon maintaining conversation context."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client: Optional[AsyncOpenAI] = None
        self.msgs: List[Dict[str, str]] = [SYSTEM_MSG]
        self.model = cfg.model

        if cfg.api_key:
            self.client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def trim_history(self) -> None:
        max_len = self.cfg.history + 1
        if len(self.msgs) > max_len:
            self.msgs = [self.msgs[0]] + self.msgs[-self.cfg.history :]

    def switch_model(self, name: str) -> None:
        self.model = name

    def clear(self) -> None:
        self.msgs = [SYSTEM_MSG]

    async def chat(self, text: str) -> AsyncGenerator[Event, None]:
        """Generator: yields events from stream."""
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

                if r:
                    if not r_started:
                        stats.on_token()
                        r_started = True
                        yield Event(EventType.OUTPUT, ("[Reasoning]\n", C.GRAY))
                    r_buf += r
                    stats.tokens += len(r) // 4
                    stats.r_tokens += len(r) // 4
                    yield Event(EventType.OUTPUT, (r, C.GRAY))

                if c:
                    if not c_started:
                        if not stats.first_token:
                            stats.on_token()
                        c_started = True
                        if r_started:
                            yield Event(EventType.OUTPUT, ("\n", ""))
                        yield Event(EventType.OUTPUT, (f"[Assistant]\nAssistant: ", ""))
                    c_buf += c
                    stats.tokens += len(c) // 4
                    stats.c_tokens += len(c) // 4
                    yield Event(EventType.OUTPUT, (c, ""))

            resp = c_buf or r_buf
            if resp:
                self.msgs.append({"role": "assistant", "content": resp})
                yield Event(EventType.OUTPUT, ("\n", ""))
                yield Event(EventType.STATS, stats)
                yield Event(EventType.DONE, None)
            else:
                yield Event(EventType.ERROR, "Empty response")

        except asyncio.CancelledError:
            resp = c_buf or r_buf
            if resp:
                self.msgs.append({"role": "assistant", "content": resp})
            raise

    async def list_models(self) -> List[str]:
        if not self.client:
            return []
        try:
            models = await self.client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            log(f"list_models error: {e}")
            return []


# ╭────────────────────────────────────────────────────────────╮
# │  Commands (All Uniform)                                    │
# ╰────────────────────────────────────────────────────────────╯


class Frontend:
    """Frontend Command: handles chat by consuming daemon's generator.

    Like a complex SQL query - iterates over streaming results.
    """

    def __init__(self, daemon: Daemon):
        self.daemon = daemon

    async def run(self, text: str) -> None:
        """Run chat - consumes daemon's generator, handles Ctrl-C."""
        try:
            async for ev in self.daemon.chat(text):
                self._render(ev)
        except asyncio.CancelledError:
            print(f"\n{C.GRAY}[Stop]{C.RESET}")
            raise

    def _render(self, ev: Event) -> None:
        match ev.type:
            case EventType.OUTPUT:
                text, color = ev.data
                if color:
                    print(f"{color}{text}{C.RESET}", end="", flush=True)
                else:
                    print(text, end="", flush=True)
            case EventType.STATS:
                print(ev.data)
            case EventType.ERROR:
                print(f"\n[Error: {ev.data}]")
            case EventType.DONE:
                pass


class ModelCommand:
    """/model - List or switch models."""

    def __init__(self, daemon: Daemon):
        self.daemon = daemon

    async def run(self, args: str = "") -> None:
        if args.strip():
            # Switch model
            self.daemon.switch_model(args.strip())
            print(f"Switched: {args.strip()}")
        else:
            # List models
            print("Fetching...")
            models = await self.daemon.list_models()
            if models:
                print(f"\nModels (current: {self.daemon.model}):")
                print("-" * 40)
                for m in sorted(models):
                    marker = " *" if m == self.daemon.model else ""
                    print(f"  {m}{marker}")
                print("-" * 40)
            else:
                print("No models available")


class ClearCommand:
    """/clear - Clear history."""

    def __init__(self, daemon: Daemon):
        self.daemon = daemon

    async def run(self, args: str = "") -> None:
        self.daemon.clear()
        print("Cleared")


class HelpCommand:
    """/help - Show help."""

    async def run(self, args: str = "") -> None:
        print("""
Commands:
  <text>          Chat with bot (default)
  /model [name]   List or switch models
  /clear          Clear history
  /help           This help
  /quit, /exit    Exit

Press Ctrl-C to interrupt chat.
        """)


class QuitCommand:
    """/quit - Exit."""

    async def run(self, args: str = "") -> None:
        print("Bye!")
        sys.exit(0)


# ╭────────────────────────────────────────────────────────────╮
# │  Shell (Router)                                            │
# ╰────────────────────────────────────────────────────────────╯


class Shell:
    """Shell - routes input to appropriate Command."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.daemon = Daemon(cfg)
        # All commands, including Frontend
        self.cmds: Dict[str, Callable] = {
            # Frontend is the default (no / prefix)
            "__default__": Frontend(self.daemon),
            # Named commands
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
        """Parse line into (command, args). Returns (Frontend, text) for non-commands."""
        if not line.startswith("/"):
            return self.cmds["__default__"], line

        parts = line.split(maxsplit=1)
        name = parts[0][1:].lower()
        args = parts[1] if len(parts) > 1 else ""

        cmd = self.cmds.get(name)
        if cmd:
            return cmd, args
        # Unknown /command - treat as chat
        return self.cmds["__default__"], line

    async def run(self) -> None:
        """Main loop."""
        print(f"🤖 {self.cfg.base_url} | {self.daemon.model}")
        print("Commands: /model, /clear, /help, /quit")
        print("Default: chat (type anything)\n")

        if not self.daemon.client:
            print("⚠️  No API key")

        while True:
            try:
                line = input(PROMPT).strip()
                if not line:
                    continue

                cmd, args = self._get_cmd(line)

                # Execute with cancellation support
                task = asyncio.create_task(cmd.run(args))
                try:
                    await task
                except asyncio.CancelledError:
                    # Task was cancelled (Ctrl-C during chat)
                    pass

            except KeyboardInterrupt:
                # Ctrl-C outside of task - just show prompt again
                print(f"\n{C.GRAY}[Interrupt]{C.RESET}")
            except EOFError:
                print("\nBye!")
                break
            except Exception as e:
                print(f"\n[Error: {e}]")
                log(f"shell error: {type(e).__name__}")


# ╭────────────────────────────────────────────────────────────╮
# │  Entry                                                     │
# ╰────────────────────────────────────────────────────────────╯


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Minimal TUI Chatbot - Shell Architecture",
        epilog="""
Examples:
  %(prog)s
  %(prog)s --api-key sk-xxx --model gpt-4
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
