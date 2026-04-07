"""
Minimal TUI Chatbot - Event-driven architecture with Shell-like logic.

Design:
    - Global unified logger
    - ChatBot as generator yielding events
    - Frontend consumes events (symmetric to commands)
    - Ctrl-C sends signal to current running operation
    - Clean separation: logic (ChatBot) vs presentation (Frontend)
"""

import os
import sys
import time
import signal
import argparse
import asyncio
from typing import List, Dict, Optional, Any, NamedTuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum, auto
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


# ╭────────────────────────────────────────────────────────────╮
# │  Global Logger (Unified)                                   │
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
            print(f"\n[DBG] {msg}", flush=True)


# Global accessor
def log(msg: str) -> None:
    Logger().log(msg)


# ╭────────────────────────────────────────────────────────────╮
# │  Constants                                                 │
# ╰────────────────────────────────────────────────────────────╯


class Style:
    """ANSI styles."""

    GRAY = "\x1b[90m"
    CYAN = "\x1b[36m"
    RESET = "\x1b[0m"


PROMPT = ">>> "
SYSTEM_MSG = {"role": "system", "content": "You are a helpful assistant."}


# ╭────────────────────────────────────────────────────────────╮
# │  Data Models                                               │
# ╰────────────────────────────────────────────────────────────╯


class EventType(Enum):
    """Stream event types - symmetric design."""

    REASONING = auto()  # Gray thinking
    CONTENT = auto()  # Normal output
    STATS = auto()  # Statistics line
    STOP = auto()  # Interrupted
    ERROR = auto()  # Error occurred


@dataclass(frozen=True)
class Event:
    """Immutable event from stream."""

    type: EventType
    data: Any = None


@dataclass(frozen=True)
class Config:
    """Bot configuration."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    debug: bool = False
    history: int = 10


@dataclass
class Stats:
    """Streaming statistics."""

    r_tokens: int = 0
    c_tokens: int = 0
    start: float = field(default_factory=time.time)
    first_token: Optional[float] = None

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    @property
    def total(self) -> int:
        return self.r_tokens + self.c_tokens

    @property
    def tps(self) -> float:
        return self.total / self.elapsed if self.elapsed > 0 else 0.0

    def on_token(self) -> None:
        if self.first_token is None:
            self.first_token = time.time()

    def __str__(self) -> str:
        total = self.total
        if total == 0:
            return "[0 | 0% | 0% | 0.0s]\n[TPS 0.0 | AVG 0.0 | TTFT 0.0s]"

        r_pct = (self.r_tokens / total) * 100
        c_pct = (self.c_tokens / total) * 100
        tps = self.tps
        ttft = self.first_token - self.start if self.first_token else 0.0

        line1 = f"[{total} | {r_pct:.1f}% | {c_pct:.1f}% | {self.elapsed:.1f}s]"
        line2 = f"[TPS {tps:.1f} | AVG {tps:.1f} | TTFT {ttft:.2f}s]"
        return f"{line1}\n{line2}"


class Chunk(NamedTuple):
    """Extracted chunk."""

    r: Optional[str] = None
    c: Optional[str] = None
    done: bool = False


# ╭────────────────────────────────────────────────────────────╮
# │  Pure Functions                                            │
# ╰────────────────────────────────────────────────────────────╯


def extract(chunk: Any) -> Chunk:
    """Pure function: extract content from API chunk."""
    if not chunk.choices:
        return Chunk()
    choice = chunk.choices[0]
    delta = choice.delta
    if not delta:
        return Chunk()
    return Chunk(
        r=getattr(delta, "reasoning_content", None),
        c=getattr(delta, "content", None),
        done=getattr(choice, "finish_reason", None) is not None,
    )


# ╭────────────────────────────────────────────────────────────╮
# │  ChatBot (Generator - Pure Logic)                         │
# ╰────────────────────────────────────────────────────────────╯


class ChatBot:
    """Pure logic: generates events from stream."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client: Optional[AsyncOpenAI] = None
        self.msgs: List[Dict[str, str]] = [SYSTEM_MSG]
        self.model = cfg.model

        if cfg.api_key:
            self.client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def trim_history(self) -> None:
        """Trim messages to max history."""
        max_len = self.cfg.history + 1
        if len(self.msgs) > max_len:
            self.msgs = [self.msgs[0]] + self.msgs[-self.cfg.history :]

    async def chat(self, text: str) -> AsyncGenerator[Event, None]:
        """Generator: yields events from stream.

        Usage:
            async for event in bot.chat("hi"):
                print(event)
        """
        if not self.client:
            yield Event(EventType.ERROR, "No API key")
            return

        self.trim_history()
        self.msgs.append({"role": "user", "content": text})

        stats = Stats()
        r_buf, c_buf = "", ""
        r_started, c_started = False, False

        try:
            log(f"stream: model={self.model}")
            stream = await self.client.chat.completions.create(
                model=self.model, messages=self.msgs, stream=True
            )

            async for ch in stream:
                ex = extract(ch)

                # Reasoning event
                if ex.r:
                    if not r_started:
                        stats.on_token()
                        r_started = True
                    r_buf += ex.r
                    stats.r_tokens += len(ex.r) // 4
                    yield Event(EventType.REASONING, ex.r)

                # Content event
                if ex.c:
                    if not c_started:
                        if not stats.first_token:
                            stats.on_token()
                        c_started = True
                    c_buf += ex.c
                    stats.c_tokens += len(ex.c) // 4
                    yield Event(EventType.CONTENT, ex.c)

                log(f"chunk: r={bool(ex.r)} c={bool(ex.c)} done={ex.done}")

            # Final stats event
            resp = c_buf or r_buf
            if resp:
                self.msgs.append({"role": "assistant", "content": resp})
                yield Event(EventType.STATS, stats)
            else:
                yield Event(EventType.ERROR, "Empty response")

        except asyncio.CancelledError:
            # Interrupted - save partial response but don't print (Shell handles it)
            resp = c_buf or r_buf
            if resp:
                self.msgs.append({"role": "assistant", "content": resp})
            raise  # Re-raise for Shell to handle


# ╭────────────────────────────────────────────────────────────╮
# │  Frontend (Event Consumer - UI Logic)                       │
# ╰────────────────────────────────────────────────────────────╯


class Frontend:
    """UI layer: consumes events from ChatBot generator.

    Symmetric to Commands - handles the 'else' branch in Shell.
    """

    def __init__(self, bot: ChatBot):
        self.bot = bot

    def _print_event(self, ev: Event) -> None:
        """Print single event based on type."""
        match ev.type:
            case EventType.REASONING:
                print(f"{Style.GRAY}{ev.data}{Style.RESET}", end="", flush=True)
            case EventType.CONTENT:
                print(ev.data, end="", flush=True)
            case EventType.STATS:
                print(f"\n{ev.data}")
            case EventType.ERROR:
                print(f"\n[Error: {ev.data}]")

    async def handle(self, text: str) -> None:
        """Handle chat input - consumes generator events."""
        r_started = False
        c_started = False

        try:
            async for ev in self.bot.chat(text):
                # Print labels on first occurrence
                if ev.type == EventType.REASONING and not r_started:
                    print(f"\n{Style.GRAY}[Reasoning]{Style.RESET} ", end="")
                    r_started = True
                elif ev.type == EventType.CONTENT and not c_started:
                    if r_started:
                        print()  # newline after reasoning
                    print(f"\n[Assistant] ", end="")
                    c_started = True

                self._print_event(ev)

        except asyncio.CancelledError:
            # Propagate cancellation to caller (Shell)
            raise


# ╭────────────────────────────────────────────────────────────╮
# │  Commands (Symmetric to Frontend)                         │
# ╰────────────────────────────────────────────────────────────╯


class Commands:
    """Shell commands - symmetric structure to Frontend."""

    def __init__(self, shell: "Shell"):
        self.shell = shell
        self.bot = shell.bot

    async def model(self, args: str) -> None:
        """/model - List or switch models."""
        parts = args.split()
        if len(parts) > 1:
            self.bot.model = parts[1]
            print(f"Switched: {parts[1]}")
        else:
            if not self.bot.client:
                print("Error: No API key")
                return
            try:
                print("Fetching...")
                models = await self.bot.client.models.list()
                print(f"\nModels (current: {self.bot.model}):")
                print("-" * 40)
                for m in sorted(models.data, key=lambda x: x.id):
                    marker = " *" if m.id == self.bot.model else ""
                    print(f"  {m.id}{marker}")
                print("-" * 40)
            except Exception as e:
                print(f"Error: {e}")

    async def clear(self, _: str) -> None:
        """/clear - Clear history."""
        self.bot.msgs = [SYSTEM_MSG]
        print("Cleared")

    async def help(self, _: str) -> None:
        """/help - Show help."""
        print("""
/model [name]  List or switch models
/clear         Clear history
/help          This help
/quit, /exit   Exit
        """)

    async def quit(self, _: str) -> None:
        """/quit - Exit shell."""
        print("Bye!")
        sys.exit(0)


# ╭────────────────────────────────────────────────────────────╮
# │  Shell (Main Loop - Router)                                 │
# ╰────────────────────────────────────────────────────────────╯


class Shell:
    """Shell-like router: commands vs chat (else branch)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.bot = ChatBot(cfg)
        self.cmds = Commands(self)
        self.frontend = Frontend(self.bot)
        self._current_task: Optional[asyncio.Task] = None

    def _get_cmd(self, line: str) -> Optional[tuple]:
        """Parse command: returns (handler, args) or None."""
        if not line.startswith("/"):
            return None

        parts = line.split(maxsplit=1)
        cmd_name = parts[0][1:].lower()
        args = parts[1] if len(parts) > 1 else ""

        handlers = {
            "model": self.cmds.model,
            "m": self.cmds.model,
            "clear": self.cmds.clear,
            "c": self.cmds.clear,
            "help": self.cmds.help,
            "h": self.cmds.help,
            "quit": self.cmds.quit,
            "q": self.cmds.quit,
            "exit": self.cmds.quit,
        }

        handler = handlers.get(cmd_name)
        return (handler, args) if handler else None

    async def run(self) -> None:
        """Main shell loop with proper cancel handling."""
        print(f"🤖 {self.cfg.base_url} | {self.bot.model}")
        print("Commands: /model, /clear, /help, /quit\n")

        if not self.bot.client:
            print("⚠️  No API key set")

        while True:
            try:
                line = input(PROMPT).strip()
                if not line:
                    continue

                cmd = self._get_cmd(line)
                if cmd:
                    handler, args = cmd
                    await handler(args)
                else:
                    # ELSE BRANCH: Chat via Frontend
                    self._current_task = asyncio.create_task(self.frontend.handle(line))
                    try:
                        await self._current_task
                    except asyncio.CancelledError:
                        print(f"\n{Style.GRAY}[Stop]{Style.RESET}")
                    finally:
                        self._current_task = None

            except KeyboardInterrupt:
                # Signal to cancel current frontend task
                if self._current_task and not self._current_task.done():
                    self._current_task.cancel()
                    try:
                        await self._current_task
                    except asyncio.CancelledError:
                        print(f"\n{Style.GRAY}[Stop]{Style.RESET}")
                    self._current_task = None
                else:
                    print(f"\n{Style.GRAY}[Interrupt]{Style.RESET}")
            except EOFError:
                print("\nBye!")
                break
            except Exception as e:
                print(f"\n[Error: {e}]")
                log(f"shell error: {type(e).__name__}")


# ╭────────────────────────────────────────────────────────────╮
# │  Entry Point                                               │
# ╰────────────────────────────────────────────────────────────╯


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal TUI Chatbot - Event-driven architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --api-key sk-xxx --model gpt-4
  %(prog)s --base-url http://localhost:11434/v1 --model llama2
        """,
    )

    parser.add_argument(
        "--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # Setup global logger
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
        sys.exit(0)


if __name__ == "__main__":
    main()
