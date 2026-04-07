"""
Minimal TUI Chatbot with streaming output and TPS statistics.
Simple shell-like interface.
"""

import os
import sys
import time
import argparse
import asyncio
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
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
        elapsed = self.elapsed
        return self.tokens / elapsed if elapsed > 0 else 0.0


class ChatBot:
    """Minimal shell-like chatbot."""

    def __init__(self, config: ChatConfig):
        self.config = config
        self.client: Optional[AsyncOpenAI] = None
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        self.current_model = config.model

        if config.api_key:
            self.client = AsyncOpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
            )

    def log(self, msg: str) -> None:
        """Print debug log if debug mode is enabled."""
        if self.config.debug:
            print(f"[DEBUG] {msg}", flush=True)

    def print_banner(self):
        """Print welcome message."""
        print(f"🤖 ChatBot | URL: {self.config.base_url} | Model: {self.current_model}")
        print("Commands: /model, /model <name>, /clear, /help, /quit")
        if self.config.debug:
            print("[Debug mode ON]")
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
        """Clear history."""
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        print("History cleared")

    def print_help(self) -> None:
        """Print help."""
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

        self.messages.append({"role": "user", "content": user_message})
        print(f"\nYou: {user_message}")

        stats = StreamStats()
        content = ""
        has_content = False
        chunk_count = 0

        # Print assistant prefix
        print("Assistant: ", end="", flush=True)

        try:
            stream = await self.client.chat.completions.create(
                model=self.current_model,
                messages=self.messages,
                stream=True,
            )

            async for chunk in stream:
                chunk_count += 1

                # Handle different API response formats
                text_content = ""

                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta:
                        # Try standard content field first
                        c = getattr(delta, "content", None)
                        if c:
                            text_content = c
                        else:
                            # Fallback to reasoning_content (for some models like doubao)
                            r = getattr(delta, "reasoning_content", None)
                            if r:
                                text_content = r

                # Print content if not empty
                if text_content:
                    has_content = True
                    content += text_content
                    stats.tokens = len(content) // 4
                    print(text_content, end="", flush=True)

            # Finish line
            if has_content:
                print()
            else:
                print("\n[No content received]")

            print(
                f"[Stats: {stats.tokens} tokens, {stats.elapsed:.2f}s, {stats.tps:.2f} tok/s]"
            )

            if content:
                self.messages.append({"role": "assistant", "content": content})

        except asyncio.CancelledError:
            print("\n[Interrupted]")
            if content:
                self.messages.append({"role": "assistant", "content": content})
        except Exception as e:
            print(f"\nError: {e}")

    async def handle_command(self, cmd: str) -> bool:
        """Handle commands. Returns False to exit."""
        parts = cmd.strip().split()
        if not parts:
            return True

        c = parts[0].lower()

        if c in ("/quit", "/exit"):
            print("Goodbye!")
            return False

        if c == "/help":
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

    async def run(self):
        """Main loop."""
        self.print_banner()

        if not self.client:
            print("Warning: OPENAI_API_KEY not set!")

        while True:
            try:
                # Simple input prompt
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
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Minimal TUI Chatbot with TPS stats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Use defaults or .env
  %(prog)s --api-key sk-xxx          # Set API key
  %(prog)s --base-url http://localhost:11434/v1 --model llama2
  %(prog)s --model gpt-4
        """,
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="API base URL (default: https://api.openai.com/v1 or OPENAI_BASE_URL env var)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", ""),
        help="API key (default: OPENAI_API_KEY env var)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        help="Model to use (default: gpt-3.5-turbo or OPENAI_MODEL env var)",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    return ChatConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        debug=args.debug,
    )


def main():
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
