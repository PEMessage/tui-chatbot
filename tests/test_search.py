"""消息搜索引擎测试."""

import pytest
from datetime import datetime

from tui_chatbot.search.engine import (
    MessageSearchEngine,
    SearchScope,
    SearchResult,
    SearchMatch,
)
from tui_chatbot.agent.types import UserMessage, AssistantMessage, TextContent
from tui_chatbot.session.models import ChatSession, SessionMetadata


class TestMessageSearchEngine:
    """测试消息搜索引擎."""

    def test_index_and_search_basic(self):
        """测试基本索引和搜索."""
        engine = MessageSearchEngine()

        # 创建测试会话
        session = self._create_test_session(
            [
                ("user", "Hello world"),
                ("assistant", "Hi there!"),
            ]
        )

        engine.index_session(session)
        result = engine.search("Hello")

        assert result.total_matches == 1
        assert result.matches[0].matched_text == "Hello"
        assert result.scope == SearchScope.ALL
        assert result.search_time_ms >= 0

    def test_search_case_insensitive(self):
        """测试大小写不敏感搜索."""
        engine = MessageSearchEngine()
        session = self._create_test_session(
            [
                ("user", "Hello World"),
            ]
        )

        engine.index_session(session)

        # 大小写不同都应匹配
        assert engine.search("hello").total_matches == 1
        assert engine.search("HELLO").total_matches == 1
        assert engine.search("Hello").total_matches == 1

    def test_search_case_sensitive(self):
        """测试大小写敏感搜索."""
        engine = MessageSearchEngine()
        session = self._create_test_session(
            [
                ("user", "Hello World"),
            ]
        )

        engine.index_session(session)

        # 大小写敏感时只有完全匹配
        assert engine.search("Hello", case_sensitive=True).total_matches == 1
        assert engine.search("hello", case_sensitive=True).total_matches == 0

    def test_search_scope_user_only(self):
        """测试仅搜索用户消息."""
        engine = MessageSearchEngine()
        session = self._create_test_session(
            [
                ("user", "user message"),
                ("assistant", "assistant message"),
            ]
        )

        engine.index_session(session)
        result = engine.search("message", scope=SearchScope.USER_ONLY)

        assert result.total_matches == 1
        assert isinstance(result.matches[0].message, UserMessage)

    def test_search_scope_assistant_only(self):
        """测试仅搜索助手消息."""
        engine = MessageSearchEngine()
        session = self._create_test_session(
            [
                ("user", "user message"),
                ("assistant", "assistant message"),
            ]
        )

        engine.index_session(session)
        result = engine.search("message", scope=SearchScope.ASSISTANT_ONLY)

        assert result.total_matches == 1
        assert isinstance(result.matches[0].message, AssistantMessage)

    def test_search_regex(self):
        """测试正则搜索."""
        engine = MessageSearchEngine()
        session = self._create_test_session(
            [
                ("user", "test123 hello"),
                ("user", "test456 world"),
            ]
        )

        engine.index_session(session)
        result = engine.search(r"test\d+", use_regex=True)

        assert result.total_matches == 2

    def test_search_invalid_regex_fallback(self):
        """测试无效正则回退到普通字符串."""
        engine = MessageSearchEngine()
        session = self._create_test_session(
            [
                ("user", "test[abc"),
            ]
        )

        engine.index_session(session)
        # 无效正则应该被当作普通字符串处理
        result = engine.search("test[abc", use_regex=True)

        assert result.total_matches == 1

    def test_search_with_context(self):
        """测试搜索结果包含上下文."""
        engine = MessageSearchEngine(context_chars=30)
        session = self._create_test_session(
            [
                ("user", "prefix Hello world suffix"),
            ]
        )

        engine.index_session(session)
        result = engine.search("Hello")

        match = result.matches[0]
        assert "prefix" in match.context_before
        assert "suffix" in match.context_after

    def test_search_max_results(self):
        """测试最大结果数限制."""
        engine = MessageSearchEngine(max_results=2)

        messages = [("user", f"message {i}") for i in range(10)]
        session = self._create_test_session(messages)

        engine.index_session(session)
        result = engine.search("message")

        assert result.total_matches == 2  # 受限于 max_results

    def test_search_multiple_sessions(self):
        """测试跨会话搜索."""
        engine = MessageSearchEngine()

        session1 = self._create_test_session(
            [
                ("user", "hello from session 1"),
            ],
            session_id="sess-1",
            title="Session 1",
        )

        session2 = self._create_test_session(
            [
                ("user", "hello from session 2"),
            ],
            session_id="sess-2",
            title="Session 2",
        )

        engine.index_session(session1)
        engine.index_session(session2)

        result = engine.search("hello")

        assert result.total_matches == 2
        session_ids = {m.session_id for m in result.matches}
        assert session_ids == {"sess-1", "sess-2"}

    def test_search_current_session(self):
        """测试在当前会话中搜索."""
        engine = MessageSearchEngine()

        # 索引多个会话
        session1 = self._create_test_session(
            [
                ("user", "hello from session 1"),
            ],
            session_id="sess-1",
        )
        session2 = self._create_test_session(
            [
                ("user", "world from session 2"),
            ],
            session_id="sess-2",
        )

        engine.index_session(session1)
        engine.index_session(session2)

        # 只在 session2 中搜索
        result = engine.search_current_session(session2, "world")

        assert result.total_matches == 1
        assert result.matches[0].session_id == "sess-2"

        # 验证原始索引未被修改
        all_result = engine.search("hello")
        assert all_result.total_matches == 1

    def test_search_assistant_with_text_content(self):
        """测试助手消息中的文本内容搜索."""
        engine = MessageSearchEngine()

        session = ChatSession(
            metadata=SessionMetadata(
                id="test-123",
                title="Test Session",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                model="gpt-4",
            )
        )

        # 添加带 TextContent 的助手消息
        assistant_msg = AssistantMessage(
            content=[TextContent(text="This is a helpful response")]
        )
        session.add_message(assistant_msg)

        engine.index_session(session)
        result = engine.search("helpful")

        assert result.total_matches == 1
        assert result.matches[0].matched_text == "helpful"

    def test_clear_index(self):
        """测试清空索引."""
        engine = MessageSearchEngine()
        session = self._create_test_session(
            [
                ("user", "hello world"),
            ]
        )

        engine.index_session(session)
        assert engine.search("hello").total_matches == 1

        engine.clear_index()
        assert engine.search("hello").total_matches == 0

    def test_index_sessions_batch(self):
        """测试批量索引会话."""
        engine = MessageSearchEngine()

        sessions = [
            self._create_test_session([("user", f"msg {i}")], session_id=f"sess-{i}")
            for i in range(3)
        ]

        engine.index_sessions(sessions)
        result = engine.search("msg")

        assert result.total_matches == 3

    def test_search_empty_content(self):
        """测试搜索空内容消息."""
        engine = MessageSearchEngine()

        session = ChatSession(
            metadata=SessionMetadata(
                id="test-123",
                title="Test Session",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                model="gpt-4",
            )
        )

        # 添加空内容的助手消息
        assistant_msg = AssistantMessage(content=[])
        session.add_message(assistant_msg)
        session.add_message(UserMessage(content="hello"))

        engine.index_session(session)
        result = engine.search("hello")

        assert result.total_matches == 1

    def _create_test_session(
        self,
        messages: list[tuple[str, str]],
        session_id: str = "test-123",
        title: str = "Test Session",
    ) -> ChatSession:
        """创建测试会话."""
        session = ChatSession(
            metadata=SessionMetadata(
                id=session_id,
                title=title,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                model="gpt-4",
            )
        )

        for role, content in messages:
            if role == "user":
                msg = UserMessage(content=content)
            else:
                msg = AssistantMessage(content=[TextContent(text=content)])
            session.add_message(msg)

        return session


class TestSearchDataClasses:
    """测试搜索相关数据类."""

    def test_search_result_creation(self):
        """测试 SearchResult 创建."""
        result = SearchResult(
            query="test",
            total_matches=5,
            matches=[],
            scope=SearchScope.ALL,
            search_time_ms=10.5,
        )

        assert result.query == "test"
        assert result.total_matches == 5
        assert result.scope == SearchScope.ALL
        assert result.search_time_ms == 10.5

    def test_search_match_creation(self):
        """测试 SearchMatch 创建."""
        msg = UserMessage(content="Hello world")
        match = SearchMatch(
            message=msg,
            session_id="sess-1",
            session_title="Test",
            matched_text="Hello",
            match_start=0,
            match_end=5,
            context_before="",
            context_after=" world",
            timestamp=datetime.now(),
        )

        assert match.matched_text == "Hello"
        assert match.session_id == "sess-1"
        assert match.match_start == 0
        assert match.match_end == 5


class TestSearchScope:
    """测试搜索范围枚举."""

    def test_search_scope_values(self):
        """测试搜索范围枚举值."""
        assert SearchScope.ALL.name == "ALL"
        assert SearchScope.USER_ONLY.name == "USER_ONLY"
        assert SearchScope.ASSISTANT_ONLY.name == "ASSISTANT_ONLY"
