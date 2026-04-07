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


@dataclass
class ChatConfig:
    """Configuration for the chatbot."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    debug: bool = False


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
        if self.reasoning_tokens > 0:
            return (
                f"[Stats: {self.total_tokens} tokens "
                f"({self.reasoning_tokens} reasoning + {self.content_tokens} content), "
                f"{self.elapsed:.2f}s]"
            )
        return f"[Stats: {self.content_tokens} tokens, {self.elapsed:.2f}s]"


class ChunkContent(NamedTuple):
    """Content extracted from a single chunk."""

    reasoning: Optional[str] = None  # Thinking/reasoning process
    content: Optional[str] = None  # Final answer content
    is_finished: bool = False  # Whether this choice is done
    finish_reason: Optional[str] = None


class ContentExtractor:
    """Extract both reasoning and content from streaming chunks."""

    def extract(self, chunk: Any) -> ChunkContent:
        """Extract both reasoning and content fields from chunk."""
        if not chunk.choices:
            return ChunkContent()

        choice = chunk.choices[0]
        delta = choice.delta
        if not delta:
            return ChunkContent()

        # Extract both fields (may both have content)
        reasoning = getattr(delta, "reasoning_content", None)
        content = getattr(delta, "content", None)

        # Check finish state
        finish_reason = getattr(choice, "finish_reason", None)
        is_finished = finish_reason is not None

        return ChunkContent(
            reasoning=reasoning if reasoning else None,
            content=content if content else None,
            is_finished=is_finished,
            finish_reason=finish_reason,
        )


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
            self.client = AsyncOpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
            )

    def log(self, msg: str) -> None:
        """Print debug log if debug mode is enabled."""
        if self.config.debug:
            print(f"\n[DEBUG] {msg}", flush=True)

    def print_banner(self) -> None:
        """Print welcome message."""
        print(f"🤖 ChatBot | URL: {self.config.base_url} | Model: {self.current_model}")
        print("Commands: /model, /model <name>, /clear, /help, /quit")
        print()

    async def list_models(self) -> None:
        """List available models."""
        if not self.client:
            print("Error: API client not initialized. Please set API key.")
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
        """Switch model."""
        self.current_model = model_name
        print(f"Switched to model: {model_name}")

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        print("History cleared")

    def print_help(self) -> None:
        """Print help message."""
        print("""
Commands:
  /model          List all available models
  /model <name>   Switch to specified model
  /clear          Clear conversation history
  /help           Show this help
  /quit, /exit    Exit

Tips:
  Just type to chat. Use Ctrl+C to stop streaming.
        """)

    def _print_reasoning(self, text: str, is_first: bool = False) -> None:
        """Print reasoning content with visual distinction."""
        if is_first:
            # Start reasoning section with gray color
            print("\n\x1b[90m[Reasoning]\x1b[0m ", end="", flush=True)
        print(f"\x1b[90m{text}\x1b[0m", end="", flush=True)

    def _print_content(self, text: str, is_first: bool = False) -> None:
        """Print final answer content."""
        if is_first:
            print("\n\x1b[0m[Assistant]\x1b[0m ", end="", flush=True)
        print(text, end="", flush=True)

    def _end_output(self) -> None:
        """End output with reset."""
        print("\x1b[0m")  # Reset colors

    async def stream_chat(self, user_message: str) -> None:
        """Stream chat response with separated reasoning and content."""
        if not self.client:
            print("Error: Cannot chat without API key.")
            return

        # Keep conversation size manageable
        if len(self.messages) > 11:
            self.messages = [self.messages[0]] + self.messages[-10:]

        self.messages.append({"role": "user", "content": user_message})

        stats = StreamStats()

        # Track state
        reasoning_started = False
        content_started = False
        reasoning_buffer = ""
        content_buffer = ""

        chunk_count = 0

        try:
            self.log(f"Starting stream with model={self.current_model}")

            stream = await self.client.chat.completions.create(
                model=self.current_model,
                messages=self.messages,
                stream=True,
            )

            async for chunk in stream:
                chunk_count += 1
                extracted = self.extractor.extract(chunk)

                # Handle reasoning content (thinking process)
                if extracted.reasoning:
                    if not reasoning_started:
                        self._print_reasoning("", is_first=True)
                        reasoning_started = True
                    self._print_reasoning(extracted.reasoning)
                    reasoning_buffer += extracted.reasoning
                    stats.reasoning_tokens += len(extracted.reasoning) // 4

                # Handle final answer content
                if extracted.content:
                    if not content_started:
                        # End reasoning section if it was started
                        if reasoning_started:
                            self._end_output()
                        self._print_content("Assistant: ", is_first=True)
                        content_started = True
                    self._print_content(extracted.content)
                    content_buffer += extracted.content
                    stats.content_tokens += len(extracted.content) // 4

                # Check finish state
                if extracted.is_finished:
                    self.log(f"Finished: reason={extracted.finish_reason}")

                # Debug
                if self.config.debug and chunk_count <= 3:
                    self.log(
                        f"Chunk {chunk_count}: reasoning={extracted.reasoning is not None}, "
                        f"content={extracted.content is not None}, finished={extracted.is_finished}"
                    )

            # Ensure proper ending
            self._end_output()

            self.log(f"Stream complete: {chunk_count} chunks")
            self.log(
                f"Reasoning: {len(reasoning_buffer)} chars, Content: {len(content_buffer)} chars"
            )

            # Combine for history (store full interaction)
            full_response = (
                reasoning_buffer + "\n\n" + content_buffer
                if reasoning_buffer
                else content_buffer
            )
            if content_buffer or reasoning_buffer:
                self.messages.append(
                    {"role": "assistant", "content": content_buffer or reasoning_buffer}
                )
                print(stats)
            else:
                print("\n[No content received]")

        except asyncio.CancelledError:
            self._end_output()
            print("\n[Interrupted]")
            if content_buffer or reasoning_buffer:
                self.messages.append(
                    {"role": "assistant", "content": content_buffer or reasoning_buffer}
                )
        except Exception as e:
            self._end_output()
            print(f"\nError: {e}")
            self.log(f"Exception: {type(e).__name__}: {e}")

    async def handle_command(self, cmd: str) -> bool:
        """Handle commands. Returns False to exit."""
        parts = cmd.strip().split()
        if not parts:
            return True

        c = parts[0].lower()

        if c in ("/quit", "/exit"):
            print("Goodbye!")
            return False
        elif c == "/help":
            self.print_help()
        elif c == "/clear":
            self.clear_history()
        elif c == "/model":
            if len(parts) > 1:
                self.switch_model(parts[1])
            else:
                await self.list_models()
        else:
            print(f"Unknown command: {c}")

        return True

    async def run(self) -> None:
        """Main chat loop."""
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
                    print("Error: No API key set.")
                    continue

                await self.stream_chat(user_input)

            except KeyboardInterrupt:
                print("\nUse /quit to exit")
            except EOFError:
                print("\nGoodbye!")
                break


def parse_args() -> ChatConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Minimal TUI Chatbot with reasoning/content separation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Use defaults or .env
  %(prog)s --api-key sk-xxx          # Set API key
  %(prog)s --base-url http://localhost:11434/v1 --model llama2
  %(prog)s --model gpt-4
  %(prog)s --debug                   # Enable debug output
        """,
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="API base URL (default: https://api.openai.com/v1)",
    )

    parser.add_argument(
        "--api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""), help="API key"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        help="Model to use",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    return ChatConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        debug=args.debug,
    )


def main() -> None:
    """Entry point."""
    config = parse_args()
    bot = ChatBot(config)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
