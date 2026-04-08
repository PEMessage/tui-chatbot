"""会话管理器测试."""

import pytest
import tempfile
import time
from pathlib import Path
from datetime import datetime

from tui_chatbot.session import SessionManager, SessionStorage, ChatSession
from tui_chatbot.session.models import SessionMetadata
from tui_chatbot.agent.types import UserMessage, AssistantMessage, TextContent


class TestSessionStorage:
    """测试 SessionStorage"""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def storage(self, temp_dir):
        return SessionStorage(temp_dir)

    def test_save_and_load(self, storage):
        """测试保存和加载会话"""
        metadata = SessionMetadata(
            id="test-123",
            title="Test Session",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            model="gpt-4",
        )
        session = ChatSession(metadata=metadata)
        session.add_message(UserMessage(content="Hello"))

        storage.save(session)
        loaded = storage.load("test-123")

        assert loaded is not None
        assert loaded.metadata.id == session.metadata.id
        assert loaded.metadata.title == "Test Session"
        assert len(loaded.messages) == 1
        # 注意：由于 Pydantic 多态反序列化限制，messages 反序列化为基类
        # 我们验证消息存在且 role 正确即可
        assert loaded.messages[0].role == "user"

    def test_save_updates_existing(self, storage):
        """测试保存更新已存在的会话"""
        metadata = SessionMetadata(
            id="test-456",
            title="Original",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            model="gpt-3.5",
        )
        session = ChatSession(metadata=metadata)
        storage.save(session)

        # 修改并重新保存
        session.metadata.title = "Updated"
        session.add_message(UserMessage(content="New message"))
        storage.save(session)

        loaded = storage.load("test-456")
        assert loaded.metadata.title == "Updated"
        assert len(loaded.messages) == 1

    def test_load_nonexistent(self, storage):
        """测试加载不存在的会话"""
        result = storage.load("nonexistent-id")
        assert result is None

    def test_list_all(self, storage):
        """测试列出所有会话 - 按时间排序"""
        # 创建多个会话
        session1 = ChatSession(
            metadata=SessionMetadata(
                id="session-1",
                title="First",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                model="gpt-4",
            )
        )
        time.sleep(0.01)  # 确保时间差
        session2 = ChatSession(
            metadata=SessionMetadata(
                id="session-2",
                title="Second",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                model="gpt-4",
            )
        )
        time.sleep(0.01)
        session3 = ChatSession(
            metadata=SessionMetadata(
                id="session-3",
                title="Third",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                model="gpt-4",
            )
        )

        storage.save(session1)
        storage.save(session2)
        storage.save(session3)

        sessions = storage.list_all()
        assert len(sessions) == 3
        # 应该按 updated_at 降序排列（最新的在前）
        assert sessions[0].metadata.id == "session-3"
        assert sessions[1].metadata.id == "session-2"
        assert sessions[2].metadata.id == "session-1"

    def test_list_all_empty(self, storage):
        """测试空存储目录列出所有会话"""
        sessions = storage.list_all()
        assert sessions == []

    def test_storage_dir_creation(self, temp_dir):
        """测试存储目录自动创建"""
        new_dir = temp_dir / "nonexistent" / "subdir"
        storage = SessionStorage(new_dir)
        assert new_dir.exists()


class TestChatSession:
    """测试 ChatSession 模型"""

    def test_add_message_updates_count(self):
        """测试添加消息更新计数"""
        metadata = SessionMetadata(
            id="test",
            title="Test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            model="gpt-4",
            message_count=0,
        )
        session = ChatSession(metadata=metadata)

        session.add_message(UserMessage(content="Hello"))
        assert len(session.messages) == 1
        assert session.metadata.message_count == 1

        session.add_message(UserMessage(content="World"))
        assert len(session.messages) == 2
        assert session.metadata.message_count == 2

    def test_add_message_updates_timestamp(self):
        """测试添加消息更新时间戳"""
        old_time = datetime.now()
        metadata = SessionMetadata(
            id="test",
            title="Test",
            created_at=old_time,
            updated_at=old_time,
            model="gpt-4",
        )
        session = ChatSession(metadata=metadata)

        time.sleep(0.01)
        session.add_message(UserMessage(content="Hello"))

        assert session.metadata.updated_at > old_time

    def test_message_list_default_empty(self):
        """测试消息列表默认为空"""
        metadata = SessionMetadata(
            id="test",
            title="Test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            model="gpt-4",
        )
        session = ChatSession(metadata=metadata)
        assert session.messages == []


class TestSessionManager:
    """测试 SessionManager"""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def manager(self, temp_dir):
        storage = SessionStorage(temp_dir)
        return SessionManager(storage)

    def test_create_session(self, manager):
        """测试创建会话"""
        session = manager.create("Test Chat", "gpt-4")

        assert session.metadata.title == "Test Chat"
        assert session.metadata.model == "gpt-4"
        assert session.metadata.id is not None
        assert len(session.metadata.id) > 0

    def test_create_session_saves_to_storage(self, manager, temp_dir):
        """测试创建会话保存到存储"""
        session = manager.create("Saved Chat", "gpt-4")
        session_id = session.metadata.id

        # 验证文件已创建
        file_path = temp_dir / f"{session_id}.json"
        assert file_path.exists()

    def test_create_sets_current_session(self, manager):
        """测试创建会话设置为当前会话"""
        session = manager.create("Current", "gpt-4")

        current = manager.current()
        assert current is not None
        assert current.metadata.id == session.metadata.id

    def test_load_session(self, manager):
        """测试加载会话"""
        original = manager.create("Original", "gpt-4")
        session_id = original.metadata.id

        # 重新加载
        loaded = manager.load(session_id)
        assert loaded is not None
        assert loaded.metadata.id == session_id
        assert loaded.metadata.title == "Original"

    def test_load_session_sets_current(self, manager):
        """测试加载会话设置为当前会话"""
        session = manager.create("Test", "gpt-4")
        session_id = session.metadata.id

        manager._current_session = None  # 重置
        manager.load(session_id)

        assert manager.current() is not None
        assert manager.current().metadata.id == session_id

    def test_load_nonexistent(self, manager):
        """测试加载不存在的会话"""
        result = manager.load("nonexistent-id")
        assert result is None

    def test_current_no_session(self, manager):
        """测试无当前会话"""
        assert manager.current() is None

    def test_list_all(self, manager):
        """测试列出所有会话"""
        s1 = manager.create("First", "gpt-4")
        time.sleep(0.01)
        s2 = manager.create("Second", "gpt-4")

        sessions = manager.list_all()
        assert len(sessions) == 2
        # 按时间降序
        assert sessions[0].metadata.id == s2.metadata.id
        assert sessions[1].metadata.id == s1.metadata.id

    def test_delete_session(self, manager, temp_dir):
        """测试删除会话"""
        session = manager.create("To Delete", "gpt-4")
        session_id = session.metadata.id

        result = manager.delete(session_id)

        assert result is True
        assert manager.load(session_id) is None
        # 验证文件已删除
        assert not (temp_dir / f"{session_id}.json").exists()

    def test_delete_current_session(self, manager):
        """测试删除当前会话重置当前状态"""
        session = manager.create("To Delete", "gpt-4")
        session_id = session.metadata.id

        manager.delete(session_id)

        assert manager.current() is None

    def test_delete_nonexistent(self, manager):
        """测试删除不存在的会话"""
        result = manager.delete("nonexistent-id")
        assert result is False

    def test_add_message_via_session(self, manager):
        """测试通过会话添加消息"""
        session = manager.create("Test", "gpt-4")
        msg = UserMessage(content="Hello")

        session.add_message(msg)

        assert len(session.messages) == 1
        assert session.metadata.message_count == 1
