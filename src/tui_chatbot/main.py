"""
Minimal TUI Chatbot with streaming output and TPS statistics.

Compatible with various OpenAI-compatible API endpoints.
Supports both reasoning and content streams with visual distinction.
"""

import os
import sys
import time
import argparse
import asyncio
from typing import List, Dict, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from contextlib import contextmanager

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


# ╭────────────────────────────────────────────────────────────╮
# │  Constants                                                 │
# ╰────────────────────────────────────────────────────────────╯


class C:
    """ANSI color codes."""

    GRAY = "\x1b[90m"
    RESET = "\x1b[0m"


class L:
    """Output labels."""

    R = "[Reasoning]"  # Reasoning
    A = "[Assistant]"  # Assistant


PROMPT = ">>> "
SYSTEM_MSG = {"role": "system", "content": "You are a helpful assistant."}


# ╭────────────────────────────────────────────────────────────╮
# │  Data Models                                               │
# ╰────────────────────────────────────────────────────────────╯


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

    r_tokens: int = 0  # reasoning
    c_tokens: int = 0  # content
    start: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    @property
    def total(self) -> int:
        return self.r_tokens + self.c_tokens

    def __str__(self) -> str:
        parts = [f"{self.total}t"]  # tokens
        if self.r_tokens:
            parts.append(f"{self.r_tokens}r+{self.c_tokens}c")
        parts.append(f"{self.elapsed:.1f}s")
        return " | ".join(parts)


class Chunk(NamedTuple):
    """Extracted chunk content."""

    r: Optional[str] = None  # reasoning
    c: Optional[str] = None  # content
    done: bool = False


# ╭────────────────────────────────────────────────────────────╮
# │  Pure Functions                                            │
# ╰────────────────────────────────────────────────────────────╯


def extract(chunk: Any) -> Chunk:
    """Extract content from streaming chunk."""
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
# │  ChatBot                                                   │
# ╰────────────────────────────────────────────────────────────╯


class ChatBot:
    """Minimal shell-like chatbot."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client: Optional[AsyncOpenAI] = None
        self.msgs: List[Dict[str, str]] = [SYSTEM_MSG]
        self.model = cfg.model

        if cfg.api_key:
            self.client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    # ── Utils ────────────────────────────────────────────────────

    def log(self, msg: str) -> None:
        if self.cfg.debug:
            print(f"\n[DBG] {msg}", flush=True)

    @contextmanager
    def _output(self):
        """Context manager for colored output."""
        try:
            yield
        finally:
            print(C.RESET, end="")

    def _show(
        self, text: str, *, label: str = "", color: str = "", end: str = ""
    ) -> None:
        """Print colored text with optional label."""
        prefix = f"\n{color}{label}{C.RESET} " if label else ""
        body = f"{color}{text}{C.RESET}" if color else text
        print(f"{prefix}{body}", end=end, flush=True)

    # ── Commands ─────────────────────────────────────────────────

    async def cmd_models(self) -> None:
        if not self.client:
            return print("Error: No API key")

        try:
            print("Fetching...")
            models = await self.client.models.list()
            print(f"\nModels (current: {self.model}):")
            print("-" * 40)
            for m in sorted(models.data, key=lambda x: x.id):
                print(f"  {m.id}{' *' if m.id == self.model else ''}")
            print("-" * 40)
        except Exception as e:
            print(f"Error: {e}")

    def cmd_switch(self, name: str) -> None:
        self.model = name
        print(f"Switched: {name}")

    def cmd_clear(self) -> None:
        self.msgs = [SYSTEM_MSG]
        print("Cleared")

    def cmd_help(self) -> None:
        print("""
/model        List models
/model <n>    Switch model
/clear        Clear history
/help         This help
/quit, /exit  Exit
        """)

    # ── Streaming ────────────────────────────────────────────────

    async def chat(self, text: str) -> None:
        if not self.client:
            return print("Error: No API key")

        # Trim history
        max_len = self.cfg.history + 1
        if len(self.msgs) > max_len:
            self.msgs = [self.msgs[0]] + self.msgs[-self.cfg.history :]

        self.msgs.append({"role": "user", "content": text})

        stats = Stats()
        r_buf, c_buf = "", ""
        r_started, c_started = False, False

        with self._output():
            try:
                self.log(f"model={self.model}")
                stream = await self.client.chat.completions.create(
                    model=self.model, messages=self.msgs, stream=True
                )

                async for ch in stream:
                    ex = extract(ch)

                    # Reasoning stream
                    if ex.r:
                        if not r_started:
                            self._show("", label=L.R, color=C.GRAY)
                            r_started = True
                        self._show(ex.r, color=C.GRAY, end="")
                        r_buf += ex.r
                        stats.r_tokens += len(ex.r) // 4

                    # Content stream
                    if ex.c:
                        if not c_started:
                            if r_started:
                                print()  # newline after reasoning
                            self._show("Assistant: ", label=L.A)
                            c_started = True
                        self._show(ex.c, end="")
                        c_buf += ex.c
                        stats.c_tokens += len(ex.c) // 4

                    self.log(f"r={bool(ex.r)} c={bool(ex.c)} done={ex.done}")

                print()  # final newline
                self.log(f"chunks: r={len(r_buf)} c={len(c_buf)}")

                # Save response
                resp = c_buf or r_buf
                if resp:
                    self.msgs.append({"role": "assistant", "content": resp})
                    print(f"[{stats}]")
                else:
                    print("[Empty]")

            except asyncio.CancelledError:
                print("\n[Stop]")
                if c_buf or r_buf:
                    self.msgs.append({"role": "assistant", "content": c_buf or r_buf})
            except Exception as e:
                print(f"\nError: {e}")
                self.log(f"exc: {type(e).__name__}")

    # ── Main Loop ────────────────────────────────────────────────

    async def handle(self, cmd: str) -> bool:
        parts = cmd.strip().split()
        if not parts:
            return True

        match parts[0].lower():
            case "/q" | "/quit" | "/exit":
                print("Bye!")
                return False
            case "/h" | "/help":
                self.cmd_help()
            case "/c" | "/clear":
                self.cmd_clear()
            case "/m" | "/model":
                if len(parts) > 1:
                    self.cmd_switch(parts[1])
                else:
                    await self.cmd_models()
            case _:
                print(f"Unknown: {parts[0]}")
        return True

    async def run(self) -> None:
        print(f"🤖 {self.cfg.base_url} | {self.model}")
        print("Commands: /model, /clear, /help, /quit\n")

        if not self.client:
            print("⚠️  No API key set")

        while True:
            try:
                inp = input(PROMPT).strip()
                if not inp:
                    continue

                if inp.startswith("/"):
                    if not await self.handle(inp):
                        break
                    continue

                if not self.client:
                    print("Error: No API key")
                    continue

                await self.chat(inp)

            except KeyboardInterrupt:
                print("\nUse /quit")
            except EOFError:
                print("\nBye!")
                break


# ╭────────────────────────────────────────────────────────────╮
# │  Entry                                                     │
# ╰────────────────────────────────────────────────────────────╯


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal TUI Chatbot",
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

    bot = ChatBot(Config(**vars(args)))

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nBye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
