"""
Minimal TUI Chatbot with streaming output and TPS statistics.
Compatible with various OpenAI-compatible API endpoints.

Architecture:
    - ContentExtractor protocol for API-specific text extraction
    - Simple adapter pattern for different response formats
    - Modern Python: type hints, dataclasses, protocols
"""

import os
import sys
import time
import argparse
import asyncio
from typing import List, Dict, Optional, Protocol, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

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

    tokens: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def tps(self) -> float:
        return self.tokens / self.elapsed if self.elapsed > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"[Stats: {self.tokens} tokens, {self.elapsed:.2f}s, {self.tps:.2f} tok/s]"
        )


class ContentExtractor(Protocol):
    """Protocol for extracting text content from streaming chunks.

    Different API endpoints may return content in different fields:
    - Standard OpenAI: delta.content
    - Doubao/Seed: delta.reasoning_content
    - Others: may have different structures
    """

    def extract(self, chunk: Any) -> Optional[str]:
        """Extract text content from a chunk. Return None if no content."""
        ...


# Built-in extractors


class StandardContentExtractor:
    """Standard OpenAI API: content in delta.content field."""

    def extract(self, chunk: Any) -> Optional[str]:
        if not chunk.choices:
            return None

        delta = chunk.choices[0].delta
        if not delta:
            return None

        content = getattr(delta, "content", None)
        return content if content else None


class ReasoningContentExtractor:
    """Doubao/Seed models: content in delta.reasoning_content field."""

    def extract(self, chunk: Any) -> Optional[str]:
        if not chunk.choices:
            return None

        delta = chunk.choices[0].delta
        if not delta:
            return None

        # Try reasoning_content first (for reasoning models)
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            return reasoning

        # Fallback to standard content field
        content = getattr(delta, "content", None)
        return content if content else None


class CompositeExtractor:
    """Try multiple extractors in order."""

    def __init__(self, extractors: List[ContentExtractor]):
        self.extractors = extractors

    def extract(self, chunk: Any) -> Optional[str]:
        for extractor in self.extractors:
            content = extractor.extract(chunk)
            if content:
                return content
        return None


# Registry of known API patterns
EXTRACTOR_REGISTRY: Dict[str, ContentExtractor] = {
    "default": StandardContentExtractor(),
    "openai": StandardContentExtractor(),
    "doubao": ReasoningContentExtractor(),
    "seed": ReasoningContentExtractor(),
    "ark": ReasoningContentExtractor(),  # ByteDance Ark platform
}


def get_extractor(base_url: str, model: str) -> ContentExtractor:
    """Get appropriate extractor based on URL and model hints.

    Args:
        base_url: API base URL for pattern matching
        model: Model name for pattern matching

    Returns:
        ContentExtractor instance for the detected API type
    """
    url_lower = base_url.lower()
    model_lower = model.lower()

    # Check URL patterns
    if any(host in url_lower for host in ["volces", "ark", "doubao", "seed"]):
        return ReasoningContentExtractor()

    # Check model name patterns
    if any(name in model_lower for name in ["doubao", "seed"]):
        return ReasoningContentExtractor()

    return StandardContentExtractor()


class ChatBot:
    """Minimal shell-like chatbot with pluggable content extraction."""

    def __init__(
        self, config: ChatConfig, extractor: Optional[ContentExtractor] = None
    ):
        self.config = config
        self.client: Optional[AsyncOpenAI] = None
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        self.current_model = config.model

        # Auto-detect or use provided extractor
        self.extractor = extractor or get_extractor(config.base_url, config.model)

        if config.api_key:
            self.client = AsyncOpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
            )

    def log(self, msg: str) -> None:
        """Print debug log if debug mode is enabled."""
        if self.config.debug:
            print(f"[DEBUG] {msg}", flush=True)

    def print_banner(self) -> None:
        """Print welcome message."""
        print(f"🤖 ChatBot | URL: {self.config.base_url} | Model: {self.current_model}")
        print("Commands: /model, /model <name>, /clear, /help, /quit")
        if self.config.debug:
            extractor_name = type(self.extractor).__name__
            print(f"[Debug: extractor={extractor_name}]")
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
        """Switch model and update extractor if needed."""
        self.current_model = model_name
        # Re-detect extractor for new model
        self.extractor = get_extractor(self.config.base_url, model_name)
        print(f"Switched to model: {model_name}")
        self.log(f"Updated extractor to {type(self.extractor).__name__}")

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

    async def stream_chat(self, user_message: str) -> None:
        """Stream chat response with TPS stats."""
        if not self.client:
            print("Error: Cannot chat without API key.")
            return

        # Keep conversation size manageable
        if len(self.messages) > 11:
            self.messages = [self.messages[0]] + self.messages[-10:]

        self.messages.append({"role": "user", "content": user_message})
        print(f"\nYou: {user_message}")

        stats = StreamStats()
        content = ""
        has_content = False
        chunk_count = 0

        print("Assistant: ", end="", flush=True)

        try:
            self.log(f"Starting stream with {type(self.extractor).__name__}")

            stream = await self.client.chat.completions.create(
                model=self.current_model,
                messages=self.messages,
                stream=True,
            )

            async for chunk in stream:
                chunk_count += 1

                # Use extractor to get content
                text = self.extractor.extract(chunk)

                if text:
                    has_content = True
                    content += text
                    stats.tokens = len(content) // 4
                    print(text, end="", flush=True)

                # Debug first few chunks
                if self.config.debug and chunk_count <= 3:
                    self.log(f"Chunk {chunk_count}: extracted={text is not None}")

            self.log(f"Stream complete: {chunk_count} chunks, {len(content)} chars")

            if has_content:
                print()
            else:
                print("\n[No content received]")

            print(stats)

            if content:
                self.messages.append({"role": "assistant", "content": content})

        except asyncio.CancelledError:
            print("\n[Interrupted]")
            if content:
                self.messages.append({"role": "assistant", "content": content})
        except Exception as e:
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
        description="Minimal TUI Chatbot with API compatibility layer",
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
    """Entry point with auto-detection of API type."""
    config = parse_args()

    # Auto-detect appropriate extractor
    extractor = get_extractor(config.base_url, config.model)

    if config.debug:
        print(f"[DEBUG] Using extractor: {type(extractor).__name__}")
        print(f"[DEBUG] Base URL: {config.base_url}")
        print(f"[DEBUG] Model: {config.model}")

    bot = ChatBot(config, extractor=extractor)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
