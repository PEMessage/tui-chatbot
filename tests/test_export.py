"""会话导出测试."""

import json
import pytest
from pathlib import Path
from datetime import datetime

from tui_chatbot.export.exporter import SessionExporter, ExportFormat
from tui_chatbot.agent.types import (
    UserMessage,
    AssistantMessage,
    TextContent,
    ToolCallContent,
    ToolResultMessage,
)
from tui_chatbot.session.models import ChatSession, SessionMetadata


class TestSessionExporter:
    """测试会话导出."""

    @pytest.fixture
    def sample_session(self):
        """创建示例会话."""
        session = ChatSession(
            metadata=SessionMetadata(
                id="sess-001",
                title="Test Chat",
                created_at=datetime(2026, 4, 9, 10, 0, 0),
                updated_at=datetime(2026, 4, 9, 10, 30, 0),
                model="gpt-4",
                message_count=2,
            )
        )
        session.add_message(UserMessage(content="Hello"))
        session.add_message(AssistantMessage(content=[TextContent(text="Hi there!")]))
        return session

    @pytest.fixture
    def complex_session(self):
        """创建包含多种消息类型的复杂会话."""
        session = ChatSession(
            metadata=SessionMetadata(
                id="sess-002",
                title="Complex Chat",
                created_at=datetime(2026, 4, 9, 11, 0, 0),
                updated_at=datetime(2026, 4, 9, 11, 30, 0),
                model="claude-3-opus",
                message_count=5,
            )
        )

        # 用户消息
        session.add_message(UserMessage(content="What's the weather?"))

        # 助手消息（带工具调用）
        session.add_message(
            AssistantMessage(
                content=[
                    TextContent(text="I'll check the weather for you."),
                    ToolCallContent(
                        id="call-1", name="get_weather", arguments={"city": "Beijing"}
                    ),
                ],
                stop_reason="tool_calls",
            )
        )

        # 工具结果
        session.add_message(
            ToolResultMessage(
                tool_call_id="call-1",
                tool_name="get_weather",
                content='{"temperature": 25, "condition": "sunny"}',
                is_error=False,
            )
        )

        # 助手最终回复
        session.add_message(
            AssistantMessage(
                content=[TextContent(text="It's 25°C and sunny in Beijing.")],
                stop_reason="end_turn",
            )
        )

        return session

    def test_export_markdown(self, tmp_path, sample_session):
        """测试 Markdown 导出."""
        exporter = SessionExporter()
        output_path = tmp_path / "test.md"

        result = exporter.export_session(
            sample_session, output_path, ExportFormat.MARKDOWN
        )

        assert result.exists()
        content = result.read_text()
        assert "# Test Chat" in content
        assert "sess-001" in content
        assert "gpt-4" in content
        assert "Hello" in content
        assert "Hi there!" in content

    def test_export_json(self, tmp_path, sample_session):
        """测试 JSON 导出."""
        exporter = SessionExporter()
        output_path = tmp_path / "test.json"

        result = exporter.export_session(sample_session, output_path, ExportFormat.JSON)

        assert result.exists()
        data = json.loads(result.read_text())
        assert data["metadata"]["id"] == "sess-001"
        assert data["metadata"]["title"] == "Test Chat"
        assert data["metadata"]["model"] == "gpt-4"
        assert len(data["messages"]) == 2

        # 验证消息结构
        user_msg = data["messages"][0]
        assert user_msg["role"] == "user"
        assert user_msg["content"] == "Hello"

        assistant_msg = data["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assert len(assistant_msg["content"]) == 1

    def test_export_convenience_method(self, tmp_path, sample_session, monkeypatch):
        """测试便捷的 export 方法."""
        # 临时修改 home 目录
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        (tmp_path / "Downloads").mkdir(exist_ok=True)

        exporter = SessionExporter()
        result = exporter.export(sample_session, ExportFormat.MARKDOWN)

        assert result.exists()
        assert result.parent.name == "Downloads"
        assert result.suffix == ".md"
        assert sample_session.metadata.id in result.name

    def test_export_json_convenience(self, tmp_path, sample_session, monkeypatch):
        """测试 JSON 便捷导出."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        (tmp_path / "Downloads").mkdir(exist_ok=True)

        exporter = SessionExporter()
        result = exporter.export(sample_session, ExportFormat.JSON)

        assert result.exists()
        assert result.suffix == ".json"

    def test_export_batch(self, tmp_path, sample_session, monkeypatch):
        """测试批量导出."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        (tmp_path / "Downloads").mkdir(exist_ok=True)

        exporter = SessionExporter()
        sessions = [sample_session, sample_session]
        paths = exporter.export_batch(sessions, ExportFormat.JSON)

        assert len(paths) == 2
        for path in paths:
            assert path.exists()
            assert path.suffix == ".json"

    def test_export_batch_markdown(self, tmp_path, sample_session, monkeypatch):
        """测试批量 Markdown 导出."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        (tmp_path / "Downloads").mkdir(exist_ok=True)

        exporter = SessionExporter()
        sessions = [sample_session]
        paths = exporter.export_batch(sessions, ExportFormat.MARKDOWN)

        assert len(paths) == 1
        assert paths[0].suffix == ".md"

    def test_export_sessions_markdown(self, tmp_path, sample_session):
        """测试批量导出多个会话到单个文件（Markdown）."""
        exporter = SessionExporter()
        output_path = tmp_path / "all.md"

        result = exporter.export_sessions(
            [sample_session, sample_session], output_path, ExportFormat.MARKDOWN
        )

        content = result.read_text()
        # 两个会话之间应有分隔符
        assert content.count("# Test Chat") == 2
        assert "---" in content

    def test_export_sessions_json(self, tmp_path, sample_session):
        """测试批量导出多个会话到单个文件（JSON）."""
        exporter = SessionExporter()
        output_path = tmp_path / "all.json"

        result = exporter.export_sessions(
            [sample_session, sample_session], output_path, ExportFormat.JSON
        )

        data = json.loads(result.read_text())
        assert len(data) == 2
        assert data[0]["metadata"]["id"] == "sess-001"

    def test_complex_session_markdown(self, tmp_path, complex_session):
        """测试复杂会话的 Markdown 导出."""
        exporter = SessionExporter()
        output_path = tmp_path / "complex.md"

        result = exporter.export_session(
            complex_session, output_path, ExportFormat.MARKDOWN
        )

        content = result.read_text()
        assert "# Complex Chat" in content
        assert "## User" in content
        assert "## Assistant" in content
        assert "## Tool Result: get_weather" in content
        assert "Tool Call: `get_weather`" in content
        assert "Beijing" in content
        assert "stop_reason" not in content  # 不在 Markdown 中显示
        assert "Stop reason: `tool_calls`" in content

    def test_complex_session_json(self, tmp_path, complex_session):
        """测试复杂会话的 JSON 导出."""
        exporter = SessionExporter()
        output_path = tmp_path / "complex.json"

        result = exporter.export_session(
            complex_session, output_path, ExportFormat.JSON
        )

        data = json.loads(result.read_text())
        assert len(data["messages"]) == 4

        # 检查工具调用消息
        assistant_with_tool = data["messages"][1]
        assert assistant_with_tool["role"] == "assistant"
        assert len(assistant_with_tool["content"]) == 2
        assert assistant_with_tool["content"][1]["type"] == "toolCall"
        assert assistant_with_tool["content"][1]["name"] == "get_weather"
        assert assistant_with_tool["stop_reason"] == "tool_calls"

        # 检查工具结果消息
        tool_result = data["messages"][2]
        assert tool_result["role"] == "tool_result"
        assert tool_result["tool_name"] == "get_weather"
        assert tool_result["is_error"] is False

    def test_export_creates_parent_dirs(self, tmp_path, sample_session):
        """测试导出时自动创建父目录."""
        exporter = SessionExporter()
        output_path = tmp_path / "nested" / "dirs" / "test.md"

        result = exporter.export_session(
            sample_session, output_path, ExportFormat.MARKDOWN
        )

        assert result.exists()
        assert result.parent.parent.exists()

    def test_error_message_in_markdown(self, tmp_path):
        """测试 Markdown 中包含错误消息."""
        session = ChatSession(
            metadata=SessionMetadata(
                id="sess-error",
                title="Error Session",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                model="gpt-4",
            )
        )
        session.add_message(
            AssistantMessage(
                content=[TextContent(text="Something went wrong")],
                error_message="API rate limit exceeded",
            )
        )

        exporter = SessionExporter()
        output_path = tmp_path / "error.md"
        result = exporter.export_session(session, output_path, ExportFormat.MARKDOWN)

        content = result.read_text()
        assert "**Error**: API rate limit exceeded" in content

    def test_tool_error_in_json(self, tmp_path):
        """测试 JSON 中包含工具错误."""
        session = ChatSession(
            metadata=SessionMetadata(
                id="sess-tool-error",
                title="Tool Error",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                model="gpt-4",
            )
        )
        session.add_message(
            ToolResultMessage(
                tool_call_id="call-err",
                tool_name="bad_tool",
                content="Error: Something failed",
                is_error=True,
            )
        )

        exporter = SessionExporter()
        output_path = tmp_path / "tool_error.json"
        result = exporter.export_session(session, output_path, ExportFormat.JSON)

        data = json.loads(result.read_text())
        assert data["messages"][0]["is_error"] is True
        assert data["messages"][0]["content"] == "Error: Something failed"
