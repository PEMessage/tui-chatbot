"""
Minimal TUI Chatbot with streaming output and TPS statistics.
Compatible with various OpenAI-compatible API endpoints.

Architecture:
    - Dual-field extraction: reasoning + content
    - Visual distinction between reasoning and answer
    - Proper finish detection via finish_reason
"""

import os
import sys
import time
import argparse
import asyncio
from typing import List, Dict, Optional, Any, NamedTuple
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


# ╭────────────────────────────────────────────────────────────╮
# │  Styles & Constants                                        │
# ╰────────────────────────────────────────────────────────────╯


class Style:
    """ANSI color codes for terminal output."""

    GRAY = "\x1b[90m"
    RESET = "\x1b[0m"


class Label:
    """Output labels for different content types."""

    REASONING = "[Reasoning]"
    ASSISTANT = "[Assistant]"


# ╭────────────────────────────────────────────────────────────╮
# │  Data Models                                               │
# ╰────────────────────────────────────────────────────────────╯


@dataclass
class ChatConfig:
    """Configuration for the chatbot."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    debug: bool = False
    max_history: int = 10  # Keep last N exchanges (+ system message)


@dataclass
class StreamStats:
    """Statistics for streaming output."""

    reasoning_tokens: int = 0
    content_tokens: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def total_tokens(self) -> int:
        return self.reasoning_tokens + self.content_tokens

    def __str__(self) -> str:
        parts = [f"{self.total_tokens} tokens"]
        if self.reasoning_tokens:
            parts.append(
                f"{self.reasoning_tokens} reasoning + {self.content_tokens} content"
            )
        parts.append(f"{self.elapsed:.2f}s")
        return " | ".join(parts)


class ChunkContent(NamedTuple):
    """Content extracted from a single streaming chunk."""

    reasoning: Optional[str] = None
    content: Optional[str] = None
    is_finished: bool = False
    finish_reason: Optional[str] = None


# ╭────────────────────────────────────────────────────────────╮
# │  Content Extraction                                        │
# ╰────────────────────────────────────────────────────────────╯


class ContentExtractor:
    """Extract reasoning and content from API streaming chunks."""

    def extract(self, chunk: Any) -> ChunkContent:
        """Extract both fields from chunk delta."""
        if not chunk.choices:
            return ChunkContent()

        choice = chunk.choices[0]
        delta = choice.delta
        if not delta:
            return ChunkContent()

        finish_reason = getattr(choice, "finish_reason", None)

        return ChunkContent(
            reasoning=getattr(delta, "reasoning_content", None),
            content=getattr(delta, "content", None),
            is_finished=finish_reason is not None,
            finish_reason=finish_reason,
        )


# ╭────────────────────────────────────────────────────────────╮
# │  ChatBot                                                   │
# ╰────────────────────────────────────────────────────────────╯


class ChatBot:
    """Minimal shell-like chatbot with dual-field extraction."""

    def __init__(self, config: ChatConfig):
        self.config = config
        self.client: Optional[AsyncOpenAI] = None
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        self.current_model = config.model
        self.extractor = ContentExtractor()

        if config.api_key:
            self.client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)

    # ── Logging ─────────────────────────────────────────────────

    def log(self, msg: str) -> None:
        if self.config.debug:
            print(f"\n[DEBUG] {msg}", flush=True)

    # ── UI Output ───────────────────────────────────────────────

    def print_banner(self) -> None:
        print(f"🤖 ChatBot | {self.config.base_url} | {self.current_model}")
        print("Commands: /model, /model <name>, /clear, /help, /quit\n")

    def _print_stream(
        self, text: str, label: str, color: str = "", is_start: bool = False
    ) -> None:
        """Unified stream printer with label and optional color."""
        if is_start:
            print(f"\n{color}{label}{Style.RESET} ", end="", flush=True)
        if color:
            print(f"{color}{text}{Style.RESET}", end="", flush=True)
        else:
            print(text, end="", flush=True)

    def _reset_output(self) -> None:
        print(Style.RESET)

    # ── Commands ────────────────────────────────────────────────

    async def list_models(self) -> None:
        if not self.client:
            print("Error: API client not initialized")
            return

        try:
            print("Fetching models...")
            models = await self.client.models.list()

            print(f"\nAvailable models (current: {self.current_model}):")
            print("-" * 50)
            for model in sorted(models.data, key=lambda m: m.id):
                marker = " *" if model.id == self.current_model else ""
                print(f"  {model.id}{marker}")
            print("-" * 50)
            print("Use /model <name> to switch")

        except Exception as e:
            print(f"Error: {e}")

    def switch_model(self, model_name: str) -> None:
        self.current_model = model_name
        print(f"Switched to model: {model_name}")

    def clear_history(self) -> None:
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        print("History cleared")

    def print_help(self) -> None:
        print("""
Commands:
  /model          List available models
  /model <name>   Switch to specified model
  /clear          Clear conversation history
  /help           Show this help
  /quit, /exit    Exit
        """)

    # ── Chat Streaming ────────────────────────────────────────────

    async def stream_chat(self, user_message: str) -> None:
        if not self.client:
            print("Error: Cannot chat without API key")
            return

        # Trim history if needed
        max_msgs = self.config.max_history + 1  # +1 for system message
        if len(self.messages) > max_msgs:
            self.messages = [self.messages[0]] + self.messages[
                -self.config.max_history :
            ]

        self.messages.append({"role": "user", "content": user_message})

        stats = StreamStats()
        reasoning_buf, content_buf = "", ""
        reasoning_started, content_started = False, False
        chunk_count = 0

        try:
            self.log(f"Streaming: model={self.current_model}")

            stream = await self.client.chat.completions.create(
                model=self.current_model,
                messages=self.messages,
                stream=True,
            )

            async for chunk in stream:
                chunk_count += 1
                extracted = self.extractor.extract(chunk)

                # Reasoning (thinking process)
                if extracted.reasoning:
                    if not reasoning_started:
                        self._print_stream(
                            "", Label.REASONING, Style.GRAY, is_start=True
                        )
                        reasoning_started = True
                    self._print_stream(extracted.reasoning, "", Style.GRAY)
                    reasoning_buf += extracted.reasoning
                    stats.reasoning_tokens += len(extracted.reasoning) // 4

                # Content (final answer)
                if extracted.content:
                    if not content_started:
                        if reasoning_started:
                            self._reset_output()
                        self._print_stream(
                            "Assistant: ", Label.ASSISTANT, is_start=True
                        )
                        content_started = True
                    self._print_stream(extracted.content, "", "")
                    content_buf += extracted.content
                    stats.content_tokens += len(extracted.content) // 4

                # Debug
                if self.config.debug and chunk_count <= 3:
                    self.log(
                        f"Chunk {chunk_count}: r={bool(extracted.reasoning)}, "
                        f"c={bool(extracted.content)}, f={extracted.is_finished}"
                    )

            self._reset_output()
            self.log(
                f"Complete: {chunk_count} chunks, "
                f"{len(reasoning_buf)}r/{len(content_buf)}c chars"
            )

            # Store response
            response = content_buf or reasoning_buf
            if response:
                self.messages.append({"role": "assistant", "content": response})
                print(f"[{stats}]")
            else:
                print("[No content received]")

        except asyncio.CancelledError:
            self._reset_output()
            print("\n[Interrupted]")
            if content_buf or reasoning_buf:
                self.messages.append(
                    {"role": "assistant", "content": content_buf or reasoning_buf}
                )
        except Exception as e:
            self._reset_output()
            print(f"\nError: {e}")
            self.log(f"Exception: {type(e).__name__}: {e}")

    # ── Main Loop ─────────────────────────────────────────────────

    async def handle_command(self, cmd: str) -> bool:
        parts = cmd.strip().split()
        if not parts:
            return True

        match parts[0].lower():
            case "/quit" | "/exit":
                print("Goodbye!")
                return False
            case "/help":
                self.print_help()
            case "/clear":
                self.clear_history()
            case "/model":
                if len(parts) > 1:
                    self.switch_model(parts[1])
                else:
                    await self.list_models()
            case _:
                print(f"Unknown command: {parts[0]}")

        return True

    async def run(self) -> None:
        self.print_banner()

        if not self.client:
            print("Warning: OPENAI_API_KEY not set!")

        while True:
            try:
                user_input = input(">>> ").strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    if not await self.handle_command(user_input):
                        break
                    continue

                if not self.client:
                    print("Error: No API key set")
                    continue

                await self.stream_chat(user_input)

            except KeyboardInterrupt:
                print("\nUse /quit to exit")
            except EOFError:
                print("\nGoodbye!")
                break


# ╭────────────────────────────────────────────────────────────╮
# │  Entry Point                                               │
# ╰────────────────────────────────────────────────────────────╯


def parse_args() -> ChatConfig:
    parser = argparse.ArgumentParser(
        description="Minimal TUI Chatbot with reasoning/content separation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Use defaults or .env
  %(prog)s --api-key sk-xxx          # Set API key
  %(prog)s --base-url http://localhost:11434/v1 --model llama2
  %(prog)s --model gpt-4 --debug
        """,
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="API base URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", ""),
        help="API key",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        help="Model to use",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    return ChatConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        debug=args.debug,
    )


def main() -> None:
    config = parse_args()
    bot = ChatBot(config)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
