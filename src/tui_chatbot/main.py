"""
Minimal TUI Chatbot - Clean Separation

Design:
    - Daemon: Pure OpenAI API forwarder (stateful context)
    - Events: Logic types (reasoning_token, content_token, done)
    - Frontend: UI decisions ([Reasoning] labels, colors)

Improvements (pi-mono inspired):
    - EventStream: async for + await result() dual interface
    - AbortController: standardized cancellation
"""

import os
import sys
import time
import asyncio
import shlex
from typing import List, Dict, Optional, Any, AsyncIterator, Generic, TypeVar, Literal
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
# │  AbortController (M3)                                      │
# ╰────────────────────────────────────────────────────────────╯


class AbortSignal:
    """Abort signal - check aborted or wait for abort."""

    def __init__(self, event: asyncio.Event, reason_ref: List[Optional[str]]):
        self._event = event
        self._reason_ref = reason_ref

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> Optional[str]:
        return self._reason_ref[0]

    async def wait(self) -> None:
        await self._event.wait()


class AbortController:
    """JavaScript-style abort controller with timeout support.

    Usage:
        ctrl = AbortController()
        # Pass ctrl.signal to async operations
        ctrl.abort("user cancelled")  # Signal abort
    """

    def __init__(self, timeout: Optional[float] = None):
        self._event = asyncio.Event()
        self._reason: List[Optional[str]] = [None]
        self._timeout_handle: Optional[asyncio.TimerHandle] = None

        if timeout is not None and timeout > 0:
            try:
                loop = asyncio.get_running_loop()
                self._timeout_handle = loop.call_later(
                    timeout, self.abort, f"timeout after {timeout}s"
                )
            except RuntimeError:
                # No running loop, can't set timeout
                pass

    @property
    def signal(self) -> AbortSignal:
        return AbortSignal(self._event, self._reason)

    def abort(self, reason: str = "aborted") -> None:
        if not self._event.is_set():
            self._reason[0] = reason
            self._event.set()

    def cancel_timeout(self) -> None:
        if self._timeout_handle:
            self._timeout_handle.cancel()
            self._timeout_handle = None


# ╭────────────────────────────────────────────────────────────╮
# │  EventStream (M1) - Generic Async Iterator + Result        │
# ╰────────────────────────────────────────────────────────────╯


T = TypeVar("T")
R = TypeVar("R")


class EventStream(Generic[T, R]):
    """Dual-interface event stream: async for + await result().

    Supports both iteration and promise-style result retrieval:
        stream = daemon.chat("hello")
        # Iterate events
        async for event in stream:
            print(event)
        # Get final result
        result = await stream.result()

    Inspired by pi-mono's EventStream pattern.
    """

    def __init__(self):
        self._queue: asyncio.Queue[T] = asyncio.Queue()
        self._done = asyncio.Event()
        self._result_future: asyncio.Future[R] = asyncio.Future()
        self._error: Optional[Exception] = None

    def push(self, event: T) -> None:
        """Push event to stream. No-op if stream is done."""
        if self._done.is_set():
            return
        self._queue.put_nowait(event)

    def end(self, result: Optional[R] = None) -> None:
        """End stream and set result (if provided)."""
        if not self._done.is_set():
            if result is not None and not self._result_future.done():
                self._result_future.set_result(result)
            elif not self._result_future.done():
                # Set empty/default result
                self._result_future.set_result(None)  # type: ignore
            self._done.set()

    def error(self, exc: Exception) -> None:
        """End stream with error."""
        if not self._done.is_set():
            self._error = exc
            if not self._result_future.done():
                self._result_future.set_exception(exc)
            self._done.set()

    async def result(self) -> R:
        """Get final result (promise-style)."""
        await self._done.wait()
        if self._error:
            raise self._error
        return await self._result_future

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        """Get next event (iteration-style)."""
        while True:
            # Check if done and queue empty
            if self._done.is_set() and self._queue.empty():
                raise StopAsyncIteration
            # Try to get from queue without blocking indefinitely
            try:
                return self._queue.get_nowait()
            except asyncio.QueueEmpty:
                # Wait a bit for new events or done signal
                try:
                    await asyncio.wait_for(self._done.wait(), timeout=0.1)
                    # Done is set, check queue one more time
                    if self._queue.empty():
                        raise StopAsyncIteration
                except asyncio.TimeoutError:
                    # Continue loop to check queue again
                    continue


# ╭────────────────────────────────────────────────────────────╮
# │  Events (Pure Logic - No UI)                                 │
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


ReasoningEffort = Literal["minimal", "low", "medium", "high"]


@dataclass(frozen=True)
class Config:
    """Configuration for TUI Chatbot (backward compatible).

    Note: New code should use tui_chatbot.config.Config directly.
    """

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    debug: bool = False
    history: int = 10
    reasoning_effort: Optional[ReasoningEffort] = None

    def to_new_config(self) -> "tui_chatbot.config.Config":
        """Convert to new Config from config module."""
        from .config import Config as NewConfig

        return NewConfig(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            debug=self.debug,
            history=self.history,
            reasoning_effort=self.reasoning_effort,
        )


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
    - EventStream (async iter + result)
    - AbortController support
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

    def chat(
        self, text: str, signal: Optional[AbortSignal] = None
    ) -> EventStream[Event, str]:
        """Generate logical events from API stream.

        Returns EventStream supporting both:
            async for event in stream: ...  # Iterate
            result = await stream.result()   # Get final message

        Args:
            text: User input
            signal: Optional abort signal for cancellation
        """
        stream = EventStream[Event, str]()

        async def _stream():
            if not self.client:
                stream.push(Event(EventType.ERROR, "No API key"))
                stream.end("")
                return

            if signal and signal.aborted:
                stream.push(Event(EventType.ERROR, f"Aborted: {signal.reason}"))
                stream.end("")
                return

            self.trim_history()
            self.msgs.append({"role": "user", "content": text})

            stats = Stats()
            r_buf, c_buf = "", ""

            try:
                log(f"stream: model={self.model}")
                create_params = {
                    "model": self.model,
                    "messages": self.msgs,
                    "stream": True,
                }
                if self.cfg.reasoning_effort:
                    create_params["reasoning_effort"] = self.cfg.reasoning_effort
                    log(f"reasoning_effort: {self.cfg.reasoning_effort}")
                api_stream = await self.client.chat.completions.create(**create_params)

                async for chunk in api_stream:
                    # Check abort signal
                    if signal and signal.aborted:
                        log(f"abort: {signal.reason}")
                        raise asyncio.CancelledError(f"Aborted: {signal.reason}")

                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue

                    r = getattr(delta, "reasoning_content", None)
                    c = getattr(delta, "content", None)

                    if r:
                        if not r_buf:
                            stats.on_token()
                        r_buf += r
                        stats.tokens += len(r) // 4
                        stats.r_tokens += len(r) // 4
                        stream.push(Event(EventType.REASONING_TOKEN, r))

                    if c:
                        if not c_buf and not r_buf:
                            stats.on_token()
                        c_buf += c
                        stats.tokens += len(c) // 4
                        stats.c_tokens += len(c) // 4
                        stream.push(Event(EventType.CONTENT_TOKEN, c))

                # Complete - save to history
                resp = c_buf or r_buf
                if resp:
                    self.msgs.append({"role": "assistant", "content": resp})
                    stream.push(Event(EventType.STATS, stats))
                    stream.push(Event(EventType.DONE, None))
                    stream.end(resp)
                else:
                    stream.push(Event(EventType.ERROR, "Empty response"))
                    stream.end("")

            except asyncio.CancelledError as e:
                # Interrupted - save partial to history
                resp = c_buf or r_buf
                if resp:
                    self.msgs.append({"role": "assistant", "content": resp})
                stream.push(Event(EventType.ERROR, f"Cancelled: {e}"))
                stream.end(resp)
                raise

            except Exception as e:
                log(f"stream error: {e}")
                stream.push(Event(EventType.ERROR, str(e)))
                stream.end("")

        # Start streaming in background task
        asyncio.create_task(_stream())
        return stream

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
# │  Commands (All Uniform)                                      │
# ╰────────────────────────────────────────────────────────────╯


class Frontend:
    """Frontend Command - renders logical events as UI.

    Decides:
    - When to print [Reasoning] label (first REASONING_TOKEN)
    - When to print [Assistant] label (first CONTENT_TOKEN)
    - Colors for reasoning vs content

    Meta:
        pass_mode: "raw" - entire input as single arg (for chat)
    """

    # Command meta info - tells Shell how to pass input
    META = {"pass_mode": "raw"}

    def __init__(self, daemon: Daemon):
        self.daemon = daemon

    async def run(self, argv: List[str]) -> None:
        """Consume events and render UI.

        For raw mode: argv[0] is the entire text.
        """
        text = argv[0] if argv else ""
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
            stream = self.daemon.chat(text)
            async for ev in stream:
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

    META = {"pass_mode": "shlex"}

    def __init__(self, daemon: Daemon):
        self.daemon = daemon

    async def run(self, argv: List[str]) -> None:
        """Run with argv like real shell: /model or /model <name>"""
        if len(argv) > 1:
            # /model <name>
            self.daemon.switch_model(argv[1])
            print(f"Switched: {argv[1]}")
        else:
            # /model (list)
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

    META = {"pass_mode": "shlex"}

    def __init__(self, daemon: Daemon):
        self.daemon = daemon

    async def run(self, argv: List[str]) -> None:
        self.daemon.clear()
        print("Cleared")


class HelpCommand:
    """/help - Show help."""

    META = {"pass_mode": "shlex"}

    async def run(self, argv: List[str]) -> None:
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

    META = {"pass_mode": "shlex"}

    async def run(self, argv: List[str]) -> None:
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
        """Parse line into (command, argv) using shlex like real shell.

        Returns: (command, argv_list)

        Pass mode is determined by command's META:
        - "raw": entire input as single arg (for chat)
        - "shlex": parse with shlex (for structured commands)
        """
        is_command = line.startswith("/")
        cmd_name = ""
        cmd = None

        if is_command:
            # Peek at command name without full parsing
            parts = line.split(None, 1)
            if parts:
                cmd_name = parts[0][1:].lower()
                cmd = self.cmds.get(cmd_name)

        if not is_command or not cmd:
            # Default: use Frontend (chat)
            cmd = self.cmds["__default__"]
            # Check command's meta for pass mode
            pass_mode = getattr(cmd, "META", {}).get("pass_mode", "raw")
            if pass_mode == "raw":
                return cmd, [line]
            else:
                # Should not happen for default, but fallback
                return cmd, shlex.split(line)

        # Known command - check META
        pass_mode = getattr(cmd, "META", {}).get("pass_mode", "shlex")

        if pass_mode == "raw":
            # Raw mode: remove command name, rest is single arg
            if " " in line:
                _, rest = line.split(None, 1)
                return cmd, [rest]
            return cmd, [""]
        else:
            # Shlex mode: full shell-like parsing
            try:
                argv = shlex.split(line)
            except ValueError:
                argv = line.split()
            return cmd, argv

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

                cmd, argv = self._get_cmd(line)
                task = asyncio.create_task(cmd.run(argv))
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


async def run_command(cfg: Config, line: str) -> int:
    """Execute a single command non-interactively.

    Returns exit code (0 for success, 1 for error).
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Minimal TUI Chatbot - Clean Separation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s                           # Interactive mode
  %(prog)s --model o3-mini --reason-effort high
  %(prog)s -c "hello world"          # Single chat message
  %(prog)s -c "/model gpt-4"         # Switch model
        """,
    )

    parser.add_argument(
        "--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--reason-effort",
        choices=["minimal", "low", "medium", "high"],
        default=os.getenv("REASONING_EFFORT"),
        help="Reasoning effort for reasoning models (o1, o3, etc.). 'minimal' maps to 'low'. Env: REASONING_EFFORT",
    )
    parser.add_argument(
        "-c",
        dest="command",
        help="Execute a single command and exit (non-interactive mode)",
    )

    args = parser.parse_args()
    Logger.enabled = args.debug

    cfg = Config(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        debug=args.debug,
        reasoning_effort=args.reason_effort,
    )

    # Non-interactive mode: execute single command
    if args.command:
        exit_code = asyncio.run(run_command(cfg, args.command))
        sys.exit(exit_code)

    # Interactive mode - use new Shell from shell module
    try:
        from .shell import create_shell
        from .config import Config as NewConfig

        new_cfg = NewConfig(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model=cfg.model,
            debug=cfg.debug,
            history=cfg.history,
            reasoning_effort=cfg.reasoning_effort,
        )
        shell = create_shell(new_cfg)
        asyncio.run(shell.run())
    except ImportError:
        # Fallback to legacy shell if new modules not available
        shell = Shell(cfg)
        try:
            asyncio.run(shell.run())
        except KeyboardInterrupt:
            print("\nBye!")


if __name__ == "__main__":
    main()
