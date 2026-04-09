"""Frontend - renders logical events as UI.

Handles both old Event types and new AssistantMessageEvent types,
converting them to appropriate UI rendering.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from .daemon import Daemon


# ANSI color codes
class Colors:
    GRAY = "\x1b[90m"
    RESET = "\x1b[0m"
    RED = "\x1b[31m"
    YELLOW = "\x1b[33m"
    GREEN = "\x1b[32m"


class Frontend:
    """Renders logical events as UI.

    Handles both old Event types (from main.py) and new
    AssistantMessageEvent types (from events.py), converting
    them to appropriate UI rendering.

    Example:
        daemon = Daemon(config)
        frontend = Frontend(daemon)
        await frontend.run(["Hello!"])
    """

    def __init__(self, daemon: Daemon):
        self.daemon = daemon

    async def run(self, argv: List[str]) -> None:
        """Consume events and render UI.

        Handles both old Event types and new AssistantMessageEvent types.

        Args:
            argv: Command arguments, argv[0] is the text to send
        """
        text = argv[0] if argv else ""

        try:
            stream = self.daemon.chat(text)
            async for event in stream:
                self._render_event(event)

            # Print newline after response
            print()

        except asyncio.CancelledError:
            print(f"{Colors.GRAY}[Cancelled]{Colors.RESET}")
            raise

    def _render_event(self, event: Any) -> None:
        """Render a single event with appropriate formatting.

        Args:
            event: Event to render (old or new style)
        """
        # Check if this is a new-style AssistantMessageEvent
        if hasattr(event, "type"):
            event_type = event.type

            # Handle new AssistantMessageEvent types
            if event_type == "start":
                # Silent - no output
                pass

            elif event_type == "text_start":
                print("[Assistant]: ", end="", flush=True)

            elif event_type == "text_delta":
                # Print delta text
                delta = getattr(event, "delta", "")
                print(delta, end="", flush=True)

            elif event_type == "text_end":
                # Silent - no output
                pass

            elif event_type == "thinking_start":
                print(f"{Colors.GRAY}[Reasoning]: {Colors.RESET}", end="", flush=True)

            elif event_type == "thinking_delta":
                # Print delta in gray
                delta = getattr(event, "delta", "")
                print(f"{Colors.GRAY}{delta}{Colors.RESET}", end="", flush=True)

            elif event_type == "thinking_end":
                # Silent - no output, but add newline
                print(Colors.RESET)

            elif event_type == "toolcall_start":
                # Get tool call name from partial if available
                partial = getattr(event, "partial", None)
                tool_name = self._extract_tool_name(
                    partial, getattr(event, "content_index", 0)
                )
                print(f"\n[Tool: {tool_name}]")

            elif event_type == "toolcall_delta":
                # Tool call arguments streaming - usually silent
                pass

            elif event_type == "toolcall_end":
                # Silent - tool execution is handled by agent
                pass

            elif event_type == "done":
                # Print stats/usage if available
                message = getattr(event, "message", None)
                if message and hasattr(message, "usage"):
                    self._render_stats(message.usage)

            elif event_type == "error":
                # Print error message
                error = getattr(event, "error", None)
                if error and hasattr(error, "errorMessage"):
                    print(f"\n{Colors.RED}[Error: {error.errorMessage}]{Colors.RESET}")
                else:
                    print(f"\n{Colors.RED}[Error]{Colors.RESET}")

            # Handle old-style Event types for backward compatibility
            elif self._is_old_event(event):
                self._render_old_event(event)

    def _extract_tool_name(self, partial: Any, content_index: int) -> str:
        """Extract tool name from partial message.

        Args:
            partial: Partial AssistantMessage
            content_index: Index of content block

        Returns:
            Tool name or "unknown"
        """
        if partial and hasattr(partial, "content"):
            content = partial.content
            if isinstance(content, list) and len(content) > content_index:
                item = content[content_index]
                if hasattr(item, "name"):
                    return item.name
        return "unknown"

    def _is_old_event(self, event: Any) -> bool:
        """Check if event is an old-style Event from main.py.

        Args:
            event: Event to check

        Returns:
            True if old-style event
        """
        # Old events have type as EventType enum
        return hasattr(event, "type") and hasattr(event.type, "name")

    def _render_old_event(self, event: Any) -> None:
        """Render old-style Event from main.py.

        Args:
            event: Old-style event with EventType enum
        """
        # Get the type name from the EventType enum
        type_name = event.type.name if hasattr(event.type, "name") else str(event.type)

        if type_name == "REASONING_TOKEN":
            # Reasoning content in gray
            data = getattr(event, "data", "")
            print(f"{Colors.GRAY}{data}{Colors.RESET}", end="", flush=True)

        elif type_name == "CONTENT_TOKEN":
            # Regular content
            data = getattr(event, "data", "")
            print(data, end="", flush=True)

        elif type_name == "STATS":
            # Statistics
            data = getattr(event, "data", None)
            if data:
                print(f"\n{data}")

        elif type_name == "ERROR":
            # Error message
            data = getattr(event, "data", "")
            print(f"\n{Colors.RED}[Error: {data}]{Colors.RESET}")

        elif type_name == "DONE":
            # Silent
            pass

    def _render_stats(self, usage: Any) -> None:
        """Render usage statistics.

        Args:
            usage: Usage object with token counts
        """
        if usage is None:
            return

        # Extract values safely
        input_tokens = getattr(usage, "inputTokens", 0)
        output_tokens = getattr(usage, "outputTokens", 0)
        total_tokens = getattr(usage, "totalTokens", 0)
        cost = getattr(usage, "cost", 0.0)

        if total_tokens > 0:
            cost_str = f" | ${cost:.4f}" if cost > 0 else ""
            print(
                f"\n{Colors.GREEN}[{total_tokens} tokens"
                f" | {input_tokens} in"
                f" | {output_tokens} out"
                f"{cost_str}]{Colors.RESET}"
            )

    def _print_labeled(self, label: str, text: str, color: str = "") -> None:
        """Print labeled text with optional color.

        Args:
            label: Label to print
            text: Text to print
            color: ANSI color code
        """
        if color:
            print(f"{color}{label}: {text}{Colors.RESET}", end="", flush=True)
        else:
            print(f"{label}: {text}", end="", flush=True)
