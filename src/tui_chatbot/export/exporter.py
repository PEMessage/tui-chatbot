"""会话导出模块 - 支持 Markdown/JSON 格式导出."""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional

from ..session.models import ChatSession
from ..agent.types import (
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
    TextContent,
    ToolCallContent,
)


class ExportFormat(Enum):
    """导出格式."""

    MARKDOWN = auto()  # Markdown 格式
    JSON = auto()  # JSON 格式


class SessionExporter:
    """会话导出器.

    支持导出为 Markdown 或 JSON 格式。

    Usage:
        >>> exporter = SessionExporter()
        >>> exporter.export_session(session, "output.md", ExportFormat.MARKDOWN)
        >>> exporter.export_session(session, "output.json", ExportFormat.JSON)
    """

    def export_session(
        self,
        session: ChatSession,
        output_path: Path,
        format: ExportFormat,
    ) -> Path:
        """导出单个会话.

        Args:
            session: 要导出的会话
            output_path: 输出文件路径
            format: 导出格式

        Returns:
            Path: 实际输出路径
        """
        if format == ExportFormat.MARKDOWN:
            content = self._to_markdown(session)
        else:
            content = self._to_json(session)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

        return output_path

    def export_sessions(
        self,
        sessions: list[ChatSession],
        output_path: Path,
        format: ExportFormat,
    ) -> Path:
        """批量导出多个会话.

        Args:
            sessions: 会话列表
            output_path: 输出文件路径
            format: 导出格式

        Returns:
            Path: 实际输出路径
        """
        if format == ExportFormat.MARKDOWN:
            parts = [self._to_markdown(s) for s in sessions]
            content = "\n\n---\n\n".join(parts)
        else:
            data = [self._session_to_dict(s) for s in sessions]
            content = json.dumps(data, indent=2, default=str, ensure_ascii=False)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

        return output_path

    def export(
        self,
        session: ChatSession,
        format: ExportFormat = ExportFormat.MARKDOWN,
        output_path: Optional[Path] = None,
    ) -> Path:
        """导出单个会话（便捷方法）.

        Args:
            session: 要导出的会话
            format: 导出格式，默认为 Markdown
            output_path: 输出文件路径，默认为 ~/Downloads/session_{id}_{timestamp}.{ext}

        Returns:
            Path: 实际输出路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{session.metadata.id}_{timestamp}"
            if format == ExportFormat.MARKDOWN:
                filename += ".md"
            else:
                filename += ".json"
            output_path = Path.home() / "Downloads" / filename

        return self.export_session(session, output_path, format)

    def export_batch(
        self,
        sessions: List[ChatSession],
        format: ExportFormat = ExportFormat.JSON,
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        """批量导出会话（便捷方法）.

        Args:
            sessions: 要导出的会话列表
            format: 导出格式，默认为 JSON
            output_dir: 输出目录，默认为 ~/Downloads/chat_exports

        Returns:
            List[Path]: 导出的文件路径列表
        """
        if output_dir is None:
            output_dir = Path.home() / "Downloads" / "chat_exports"

        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []

        for session in sessions:
            path = output_dir / f"session_{session.metadata.id}.json"
            if format == ExportFormat.MARKDOWN:
                path = path.with_suffix(".md")
            paths.append(self.export(session, format, path))

        return paths

    def _to_markdown(self, session: ChatSession) -> str:
        """转换为 Markdown 格式."""
        lines = []

        # 标题
        lines.append(f"# {session.metadata.title}")
        lines.append("")

        # 元信息
        lines.append(f"- **Session ID**: {session.metadata.id}")
        lines.append(f"- **Model**: {session.metadata.model}")
        lines.append(f"- **Created**: {session.metadata.created_at.isoformat()}")
        lines.append(f"- **Updated**: {session.metadata.updated_at.isoformat()}")
        lines.append(f"- **Messages**: {session.metadata.message_count}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # 消息内容
        for msg in session.messages:
            lines.extend(self._format_message_md(msg))
            lines.append("")

        return "\n".join(lines)

    def _format_message_md(self, message) -> list[str]:
        """格式化单个消息为 Markdown."""
        lines = []

        if isinstance(message, UserMessage):
            lines.append(f"## User ({message.timestamp.isoformat()})")
            lines.append("")
            lines.append(message.content)

        elif isinstance(message, AssistantMessage):
            lines.append(f"## Assistant ({message.timestamp.isoformat()})")
            lines.append("")

            for content in message.content:
                if isinstance(content, TextContent):
                    lines.append(content.text)
                elif isinstance(content, ToolCallContent):
                    lines.append(f"> Tool Call: `{content.name}`")
                    lines.append(f"> ID: `{content.id}`")
                    if content.arguments:
                        args_str = json.dumps(content.arguments, ensure_ascii=False)
                        lines.append(f"> Arguments: `{args_str}`")

            if message.error_message:
                lines.append("")
                lines.append(f"> **Error**: {message.error_message}")

            if message.stop_reason:
                lines.append("")
                lines.append(f"> Stop reason: `{message.stop_reason}`")

        elif isinstance(message, ToolResultMessage):
            lines.append(f"## Tool Result: {message.tool_name or 'Unknown'}")
            lines.append("")
            lines.append(f"- Tool Call ID: `{message.tool_call_id}`")
            if message.is_error:
                lines.append("- Status: **Error**")
            lines.append("")
            lines.append("```")
            lines.append(message.content)
            lines.append("```")

        return lines

    def _to_json(self, session: ChatSession) -> str:
        """转换为 JSON 格式."""
        data = self._session_to_dict(session)
        return json.dumps(data, indent=2, default=str, ensure_ascii=False)

    def _session_to_dict(self, session: ChatSession) -> dict:
        """将会话转换为字典."""
        return {
            "metadata": {
                "id": session.metadata.id,
                "title": session.metadata.title,
                "created_at": session.metadata.created_at.isoformat(),
                "updated_at": session.metadata.updated_at.isoformat(),
                "model": session.metadata.model,
                "message_count": session.metadata.message_count,
            },
            "messages": [self._message_to_dict(m) for m in session.messages],
        }

    def _message_to_dict(self, message) -> dict:
        """将消息转换为字典."""
        base = {
            "role": getattr(message, "role", "unknown"),
            "timestamp": message.timestamp.isoformat(),
        }

        if isinstance(message, UserMessage):
            base["content"] = message.content

        elif isinstance(message, AssistantMessage):
            base["content"] = [self._content_to_dict(c) for c in message.content]
            if message.stop_reason:
                base["stop_reason"] = message.stop_reason
            if message.error_message:
                base["error"] = message.error_message

        elif isinstance(message, ToolResultMessage):
            base["tool_call_id"] = message.tool_call_id
            base["tool_name"] = message.tool_name
            base["content"] = message.content
            base["is_error"] = message.is_error

        return base

    def _content_to_dict(self, content) -> dict:
        """将内容块转换为字典."""
        if isinstance(content, TextContent):
            return {
                "type": "text",
                "text": content.text,
            }
        elif isinstance(content, ToolCallContent):
            return {
                "type": "toolCall",
                "id": content.id,
                "name": content.name,
                "arguments": content.arguments,
            }
        return {"type": "unknown"}
