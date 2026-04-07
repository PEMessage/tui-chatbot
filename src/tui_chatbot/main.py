"""
Minimal TUI Chatbot - Clean Separation

Design:
    - Daemon: Pure OpenAI API forwarder (stateful context)
    - Events: Logic types (reasoning_token, content_token, done)
    - Frontend: UI decisions ([Reasoning] labels, colors)
"""

import os
import sys
import time
import asyncio
from typing import List, Dict, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum, auto

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


# ╭────────────────────────────────────────────────────────────╮
# │  Utils                                                     │
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


class C:
    GRAY = "\x1b[90m"
    RESET = "\x1b[0m"


# ╭────────────────────────────────────────────────────────────╮
# │  Events (Pure Logic - No UI)                               │
# ╰────────────────────────────────────────────────────────────╯


class EventType(Enum):
    """Logical event types from API stream.

    No UI labels here - just what the API provides.
    """

    REASONING_TOKEN = auto()  # reasoning_content chunk
    CONTENT_TOKEN = auto()  # content chunk
    STATS = auto()  # Final statistics
    DONE = auto()  # Stream complete
    ERROR = auto()  # Error occurred


@dataclass(frozen=True)
class Event:
    """Immutable event with logical type."""

    type: EventType
    data: Any = None  # Token text, stats, or error message


@dataclass
class Stats:
    """Statistics for streaming."""

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
            return "[0 TOKENS | 0% REASONING | 0% CONTENT | 0.0s]\n[TPS 0.0 | AVG 0.0 | TTFT 0.0s]"
        r_pct = (self.r_tokens / self.tokens) * 100
        c_pct = (self.c_tokens / self.tokens) * 100
        ttft = self.first_token - self.start if self.first_token else 0.0
        return (
            f"[{self.tokens} TOKENS | {r_pct:.1f}% REASONING | {c_pct:.1f}% CONTENT | {self.elapsed:.1f}s]\n"
            f"[TPS {self.tps:.1f} | AVG {self.tps:.1f} | TTFT {ttft:.2f}s]"
        )


# ╭────────────────────────────────────────────────────────────╮
# │  Config                                                    │
# ╰────────────────────────────────────────────────────────────╯


@dataclass(frozen=True)
class Config:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    debug: bool = False
    history: int = 10


SYSTEM_MSG = {"role": "system", "content": "You are a helpful assistant."}


# ╭────────────────────────────────────────────────────────────╮
# │  Daemon (Pure API Forwarder + State)                       │
# ╰────────────────────────────────────────────────────────────╯


class Daemon:
    """Stateful daemon - forwards API stream as logical events.

    Like OpenAI client but with:
    - Context memory (msgs)
    - Trim history
    - Stats tracking
    """

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
        """Generate logical events from API stream.

        Events are pure data:
        - REASONING_TOKEN: raw reasoning text chunk
        - CONTENT_TOKEN: raw content text chunk
        - STATS: final statistics
        - DONE: stream finished successfully
        - ERROR: something went wrong
        """
        if not self.client:
            yield Event(EventType.ERROR, "No API key")
            return

        self.trim_history()
        self.msgs.append({"role": "user", "content": text})

        stats = Stats()
        r_buf, c_buf = "", ""

        try:
            log(f"stream: model={self.model}")
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
                    if not r_buf:  # First reasoning token
                        stats.on_token()
                    r_buf += r
                    stats.tokens += len(r) // 4
                    stats.r_tokens += len(r) // 4
                    yield Event(EventType.REASONING_TOKEN, r)  # Pure text, no label

                if c:
                    if not c_buf and not r_buf:  # First content, no reasoning before
                        stats.on_token()
                    c_buf += c
                    stats.tokens += len(c) // 4
                    stats.c_tokens += len(c) // 4
                    yield Event(EventType.CONTENT_TOKEN, c)  # Pure text, no label

            # Complete - save to history
            resp = c_buf or r_buf
            if resp:
                self.msgs.append({"role": "assistant", "content": resp})
                yield Event(EventType.STATS, stats)
                yield Event(EventType.DONE, None)
            else:
                yield Event(EventType.ERROR, "Empty response")

        except asyncio.CancelledError:
            # Interrupted - save partial to history
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
    """Frontend Command - renders logical events as UI.

    Decides:
    - When to print [Reasoning] label (first REASONING_TOKEN)
    - When to print [Assistant] label (first CONTENT_TOKEN)
    - Colors for reasoning vs content
    """

    def __init__(self, daemon: Daemon):
        self.daemon = daemon

    async def run(self, text: str) -> None:
        """Consume events and render UI."""
        prev_type = None
        r_started = False
        c_started = False

        # Events that produce output (not DONE which is silent)
        OUTPUT_EVENTS = {
            EventType.REASONING_TOKEN,
            EventType.CONTENT_TOKEN,
            EventType.STATS,
            EventType.ERROR,
        }

        try:
            async for ev in self.daemon.chat(text):
                # Add two newlines on type switch to output event (except first)
                if (
                    prev_type is not None
                    and ev.type != prev_type
                    and ev.type in OUTPUT_EVENTS
                ):
                    print("\n")
                prev_type = ev.type

                match ev.type:
                    case EventType.REASONING_TOKEN:
                        if not r_started:
                            print(f"{C.GRAY}[Reasoning]: {C.RESET}", end="", flush=True)
                            r_started = True
                        print(f"{C.GRAY}{ev.data}{C.RESET}", end="", flush=True)

                    case EventType.CONTENT_TOKEN:
                        if not c_started:
                            print(f"[Assistant]: ", end="")
                            c_started = True
                        print(ev.data, end="", flush=True)

                    case EventType.STATS:
                        print(f"{ev.data}")

                    case EventType.ERROR:
                        print(f"[Error: {ev.data}]")

                    case EventType.DONE:
                        pass  # Silent, no output

        except asyncio.CancelledError:
            print(f"{C.GRAY}[Stop]{C.RESET}")
            raise


class ModelCommand:
    """/model - List or switch models."""

    def __init__(self, daemon: Daemon):
        self.daemon = daemon

    async def run(self, args: str = "") -> None:
        if args.strip():
            self.daemon.switch_model(args.strip())
            print(f"Switched: {args.strip()}")
        else:
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
    """Shell - routes to Commands uniformly."""

    PROMPT = ">>> "

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.daemon = Daemon(cfg)
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
        """Parse line into (command, args)."""
        if not line.startswith("/"):
            return self.cmds["__default__"], line

        parts = line.split(maxsplit=1)
        name = parts[0][1:].lower()
        args = parts[1] if len(parts) > 1 else ""

        cmd = self.cmds.get(name)
        return (cmd, args) if cmd else (self.cmds["__default__"], line)

    async def run(self) -> None:
        """Main loop."""
        print(f"🤖 {self.cfg.base_url} | {self.daemon.model}")
        print("Commands: /model, /clear, /help, /quit")
        print("Default: chat (type anything)\n")

        if not self.daemon.client:
            print("⚠️  No API key")

        while True:
            try:
                line = input(self.PROMPT).strip()
                if not line:
                    continue

                cmd, args = self._get_cmd(line)
                task = asyncio.create_task(cmd.run(args))
                try:
                    await task
                except asyncio.CancelledError:
                    pass  # Ctrl-C handled

            except KeyboardInterrupt:
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
        description="Minimal TUI Chatbot - Clean Separation",
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
