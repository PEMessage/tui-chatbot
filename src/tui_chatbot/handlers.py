"""事件处理器 - 将事件处理逻辑从 Frontend 分离."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from .core.events import AgentEvent, AgentEventType

if TYPE_CHECKING:
    from .frontend import Frontend


class EventHandler(ABC):
    """事件处理器基类."""

    @abstractmethod
    def can_handle(self, event: AgentEvent) -> bool:
        """检查是否能处理该事件."""
        pass

    @abstractmethod
    def handle(self, event: AgentEvent, frontend: "Frontend") -> None:
        """处理事件."""
        pass


class MessageStartHandler(EventHandler):
    """处理 MESSAGE_START 事件."""

    def can_handle(self, event: AgentEvent) -> bool:
        return event.type == AgentEventType.MESSAGE_START

    def handle(self, event: AgentEvent, frontend: "Frontend") -> None:
        from .indicator import StreamingIndicator

        if frontend._indicator:
            frontend._indicator.stop()
        frontend._indicator = StreamingIndicator()
        frontend._indicator.start()
        frontend._reset_render_state()


class MessageEndHandler(EventHandler):
    """处理 MESSAGE_END 事件."""

    def can_handle(self, event: AgentEvent) -> bool:
        return event.type == AgentEventType.MESSAGE_END

    def handle(self, event: AgentEvent, frontend: "Frontend") -> None:
        if frontend._indicator:
            frontend._indicator.stop()
            frontend._indicator = None
        frontend._finalize_output()
        frontend._reset_render_state()


class ReasoningTokenHandler(EventHandler):
    """处理 REASONING_TOKEN 事件."""

    def can_handle(self, event: AgentEvent) -> bool:
        return event.type == AgentEventType.REASONING_TOKEN

    def handle(self, event: AgentEvent, frontend: "Frontend") -> None:
        if frontend._indicator:
            frontend._indicator.on_token()
        frontend.on_token(event.data, is_reasoning=True)


class ContentTokenHandler(EventHandler):
    """处理 CONTENT_TOKEN 事件."""

    def can_handle(self, event: AgentEvent) -> bool:
        return event.type == AgentEventType.CONTENT_TOKEN

    def handle(self, event: AgentEvent, frontend: "Frontend") -> None:
        if frontend._indicator:
            frontend._indicator.on_token()
        frontend.on_token(event.data, is_reasoning=False)


class MessageUpdateHandler(EventHandler):
    """处理 MESSAGE_UPDATE 事件."""

    def can_handle(self, event: AgentEvent) -> bool:
        return event.type == AgentEventType.MESSAGE_UPDATE

    def handle(self, event: AgentEvent, frontend: "Frontend") -> None:
        if frontend._indicator and hasattr(event, "token"):
            frontend._indicator.on_token()
        frontend._handle_message_update(event)


class ToolStartHandler(EventHandler):
    """处理 TOOL_EXECUTION_START 事件."""

    def can_handle(self, event: AgentEvent) -> bool:
        return event.type == AgentEventType.TOOL_EXECUTION_START

    def handle(self, event: AgentEvent, frontend: "Frontend") -> None:
        frontend.on_tool_start(event.tool_name or "unknown", event.args)


class ToolEndHandler(EventHandler):
    """处理 TOOL_EXECUTION_END 事件."""

    def can_handle(self, event: AgentEvent) -> bool:
        return event.type == AgentEventType.TOOL_EXECUTION_END

    def handle(self, event: AgentEvent, frontend: "Frontend") -> None:
        frontend.on_tool_end(
            event.tool_name or "unknown",
            event.result,
            event.is_error,
        )


class StatsHandler(EventHandler):
    """处理 STATS 事件."""

    def can_handle(self, event: AgentEvent) -> bool:
        return event.type == AgentEventType.STATS

    def handle(self, event: AgentEvent, frontend: "Frontend") -> None:
        frontend._handle_stats(event)


class ErrorHandler(EventHandler):
    """处理 ERROR 事件."""

    def can_handle(self, event: AgentEvent) -> bool:
        return event.type == AgentEventType.ERROR

    def handle(self, event: AgentEvent, frontend: "Frontend") -> None:
        if frontend._indicator:
            frontend._indicator.stop()
            frontend._indicator = None
        frontend._handle_error(event)


class SilentHandler(EventHandler):
    """处理静默事件 (AGENT_START, AGENT_END, TURN_START, TURN_END)."""

    SILENT_TYPES = {
        AgentEventType.AGENT_START,
        AgentEventType.AGENT_END,
        AgentEventType.TURN_START,
        AgentEventType.TURN_END,
    }

    def can_handle(self, event: AgentEvent) -> bool:
        return event.type in self.SILENT_TYPES

    def handle(self, event: AgentEvent, frontend: "Frontend") -> None:
        pass


class UnknownHandler(EventHandler):
    """处理未知事件类型."""

    def can_handle(self, event: AgentEvent) -> bool:
        return True  # 作为兜底处理器

    def handle(self, event: AgentEvent, frontend: "Frontend") -> None:
        from .frontend import log

        log(f"Unknown event type: {event.type}")
