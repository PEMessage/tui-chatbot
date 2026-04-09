"""Shell - routes to Commands uniformly.

Command routing and REPL (Read-Eval-Print Loop) for the TUI chatbot.
Provides interactive and non-interactive command execution.
"""

from __future__ import annotations

import asyncio
import shlex
import sys
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .config import Config
from .daemon import Daemon
from .frontend import Colors, Frontend

if TYPE_CHECKING:
    from .core.abort_controller import AbortSignal


class Command:
    """Base class for shell commands.

    All commands must implement the run method.
    Commands can specify their pass mode via META dict.
    """

    META = {"pass_mode": "shlex"}

    async def run(self, argv: List[str]) -> int:
        """Execute the command.

        Args:
            argv: Command arguments

        Returns:
            Exit code (0 for success, non-zero for errors)
        """
        raise NotImplementedError("Command must implement run()")


class DefaultCommand(Command):
    """Default command - chat with the bot.

    In 'raw' pass mode, argv[0] contains the entire text.
    """

    META = {"pass_mode": "raw"}

    def __init__(self, daemon: Daemon):
        self.daemon = daemon
        self.frontend = Frontend(daemon)

    async def run(self, argv: List[str]) -> int:
        """Run chat with the bot.

        Args:
            argv: Arguments with text in argv[0]

        Returns:
            Exit code (0 for success)
        """
        await self.frontend.run(argv)
        return 0


class ModelCommand(Command):
    """/model - List or switch models."""

    def __init__(self, daemon: Daemon):
        self.daemon = daemon

    async def run(self, argv: List[str]) -> int:
        """Run model command.

        Args:
            argv: Arguments - empty to list, or ["/model", "model_name"] to switch

        Returns:
            Exit code (0 for success)
        """
        if len(argv) > 1:
            # /model <name> - switch model
            model_name = argv[1]
            self.daemon.switch_model(model_name)
            print(f"Switched to: {model_name}")
        else:
            # /model - list available models
            print("Fetching models...")
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

        return 0


class ClearCommand(Command):
    """/clear - Clear conversation history."""

    def __init__(self, daemon: Daemon):
        self.daemon = daemon

    async def run(self, argv: List[str]) -> int:
        """Clear conversation history.

        Args:
            argv: Command arguments (ignored)

        Returns:
            Exit code (0 for success)
        """
        self.daemon.clear()
        print("Conversation history cleared")
        return 0


class HelpCommand(Command):
    """/help - Show help."""

    async def run(self, argv: List[str]) -> int:
        """Show help message.

        Args:
            argv: Command arguments (ignored)

        Returns:
            Exit code (0 for success)
        """
        print("""
Commands:
  <text>          Chat with bot (default command)
  /model [name]   List available models or switch to a model
  /clear          Clear conversation history
  /help           Show this help message
  /quit, /exit    Exit the chatbot

Press Ctrl-C to interrupt chat.
        """)
        return 0


class QuitCommand(Command):
    """/quit, /exit - Exit the chatbot."""

    async def run(self, argv: List[str]) -> int:
        """Exit the chatbot.

        Args:
            argv: Command arguments (ignored)

        Returns:
            Exit code (always 0, exits process)
        """
        print("Bye!")
        sys.exit(0)
        return 0


class Shell:
    """Shell - routes to Commands uniformly.

    Provides a REPL for interactive use, with command routing
    and proper signal handling.
    """

    PROMPT = ">>> "

    def __init__(self, config: Config):
        self.config = config
        self.daemon = Daemon(config)
        self.cmds: Dict[str, Command] = {
            "__default__": DefaultCommand(self.daemon),
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

    def _get_cmd(self, line: str) -> Tuple[Command, List[str]]:
        """Parse line into (command, argv) using shlex.

        Args:
            line: Input line from user

        Returns:
            Tuple of (command instance, argv list)

        Pass mode is determined by command's META:
        - "raw": entire input as single arg (for chat)
        - "shlex": parse with shlex (for structured commands)
        """
        is_command = line.startswith("/")
        cmd_name = ""
        cmd: Optional[Command] = None

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
        """Main REPL loop.

        Continuously prompts for input and routes commands.
        Handles Ctrl-C gracefully.
        """
        # Print startup message
        print(f"🤖 {self.config.base_url} | {self.daemon.model}")
        print("Commands: /model, /clear, /help, /quit")
        print("Default: chat (type anything)\n")

        if not self.config.api_key:
            print(f"{Colors.YELLOW}⚠️  No API key configured{Colors.RESET}")

        while True:
            try:
                line = input(self.PROMPT).strip()
                if not line:
                    continue

                cmd, argv = self._get_cmd(line)

                # Run command as async task
                task = asyncio.create_task(cmd.run(argv))
                try:
                    await task
                except asyncio.CancelledError:
                    # Ctrl-C during command execution
                    pass

            except KeyboardInterrupt:
                print(f"\n{Colors.GRAY}[Interrupt]{Colors.RESET}")
            except EOFError:
                print("\nBye!")
                break
            except Exception as e:
                print(f"\n{Colors.RED}[Error: {e}]{Colors.RESET}")

    async def run_command(self, line: str) -> int:
        """Execute a single command and return exit code.

        For non-interactive mode.

        Args:
            line: Command line to execute

        Returns:
            Exit code (0 for success, non-zero for errors)
        """
        cmd, argv = self._get_cmd(line)
        return await cmd.run(argv)


def create_shell(config: Config) -> Shell:
    """Create a new Shell instance.

    Args:
        config: Chatbot configuration

    Returns:
        Configured Shell instance
    """
    return Shell(config)
