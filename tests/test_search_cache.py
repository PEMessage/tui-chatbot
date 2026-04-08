"""测试搜索缓存功能."""

import pytest
from datetime import datetime, timedelta

from tui_chatbot.search.cache import CachedSearchEngine, CacheEntry
from tui_chatbot.search.engine import SearchScope, SearchResult
from tui_chatbot.session.models import ChatSession, SessionMetadata
from tui_chatbot.agent.types import UserMessage, AssistantMessage, TextContent


class TestCacheEntry:
    """测试缓存条目."""

    def test_cache_entry_creation(self):
        """测试缓存条目创建."""
        now = datetime.now()
        messages = []
        entry = CacheEntry(
            messages=messages,
            session_updated_at=now,
            created_at=now,
        )
        assert entry.messages == messages
        assert entry.session_updated_at == now
        assert entry.created_at == now


class TestCachedSearchEngine:
    """测试带缓存的搜索引擎."""

    def _create_test_session(self, message_count: int = 3) -> ChatSession:
        """创建测试会话."""
        now = datetime.now()
        metadata = SessionMetadata(
            id="test-session-1",
            title="Test Session",
            created_at=now,
            updated_at=now,
            model="gpt-4",
            message_count=message_count,
        )

        messages = []
        for i in range(message_count):
            if i % 2 == 0:
                msg = UserMessage(
                    content=f"User message {i} with keyword",
                    timestamp=now,
                )
            else:
                msg = AssistantMessage(
                    content=[TextContent(text=f"Assistant reply {i} with keyword")],
                    timestamp=now,
                )
            messages.append(msg)

        session = ChatSession(metadata=metadata, messages=messages)
        return session

    def test_cache_key_generation(self):
        """测试缓存键生成."""
        engine = CachedSearchEngine()
        session = self._create_test_session()

        cache_key = engine._get_cache_key(session)

        assert session.metadata.id in cache_key
        assert session.metadata.updated_at.isoformat() in cache_key

    def test_cache_miss_on_first_index(self):
        """首次索引应该缓存未命中."""
        engine = CachedSearchEngine()
        session = self._create_test_session()

        engine.index_session(session)

        assert engine._cache_misses == 1
        assert engine._cache_hits == 0
        assert len(engine._session_cache) == 1

    def test_cache_hit_on_same_session(self):
        """相同会话应使用缓存."""
        engine = CachedSearchEngine()
        session = self._create_test_session()

        # 第一次索引
        engine.index_session(session)
        assert engine._cache_misses == 1

        # 清空索引但保留缓存
        engine.clear_index()

        # 第二次索引（应命中缓存）
        engine.index_session(session)
        assert engine._cache_hits == 1
        assert engine._cache_misses == 1

    def test_cache_invalidate_on_update(self):
        """会话更新后缓存应失效."""
        engine = CachedSearchEngine()
        session = self._create_test_session()

        # 首次索引
        engine.index_session(session)
        assert engine._cache_misses == 1

        # 模拟会话更新
        session.metadata.updated_at = datetime.now() + timedelta(seconds=1)

        # 清空索引
        engine.clear_index()

        # 重新索引应未命中缓存（因为时间戳变了）
        engine.index_session(session)
        assert engine._cache_misses == 2
        assert engine._cache_hits == 0

    def test_search_with_cache(self):
        """测试使用缓存的搜索."""
        engine = CachedSearchEngine()
        session = self._create_test_session()

        # 索引并搜索
        engine.index_session(session)
        result = engine.search("keyword")

        assert result.total_matches > 0
        assert result.query == "keyword"

    def test_search_scope_filtering(self):
        """测试搜索范围过滤."""
        engine = CachedSearchEngine()
        session = self._create_test_session(message_count=4)

        engine.index_session(session)

        # 搜索所有消息
        result_all = engine.search("keyword", scope=SearchScope.ALL)

        # 仅搜索用户消息
        result_user = engine.search("keyword", scope=SearchScope.USER_ONLY)

        # 仅搜索助手消息
        result_assistant = engine.search("keyword", scope=SearchScope.ASSISTANT_ONLY)

        # 验证范围过滤生效
        assert (
            result_all.total_matches
            == result_user.total_matches + result_assistant.total_matches
        )

    def test_cache_stats(self):
        """测试缓存统计."""
        engine = CachedSearchEngine()
        session = self._create_test_session()

        # 初始状态
        stats = engine.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["cached_sessions"] == 0

        # 索引后
        engine.index_session(session)
        stats = engine.get_cache_stats()
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.0
        assert stats["cached_sessions"] == 1

        # 命中缓存
        engine.clear_index()
        engine.index_session(session)
        stats = engine.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_size_limit(self):
        """测试缓存大小限制."""
        engine = CachedSearchEngine(max_cache_size=2)

        # 创建3个不同会话
        sessions = []
        for i in range(3):
            now = datetime.now() + timedelta(seconds=i)
            metadata = SessionMetadata(
                id=f"session-{i}",
                title=f"Session {i}",
                created_at=now,
                updated_at=now,
                model="gpt-4",
            )
            msg = UserMessage(content=f"Message {i}")
            session = ChatSession(metadata=metadata, messages=[msg])
            sessions.append(session)

        # 索引3个会话（超过限制）
        for session in sessions:
            engine.index_session(session)

        # 应该只保留最近的2个
        assert len(engine._session_cache) == 2

    def test_force_reindex(self):
        """测试强制重新索引."""
        engine = CachedSearchEngine()
        session = self._create_test_session()

        # 首次索引
        engine.index_session(session)
        assert engine._cache_misses == 1

        # 强制重新索引
        engine.index_session(session, force=True)
        assert engine._cache_misses == 2
        assert engine._cache_hits == 0

    def test_clear_cache(self):
        """测试清空缓存."""
        engine = CachedSearchEngine()
        session = self._create_test_session()

        engine.index_session(session)
        assert len(engine._session_cache) == 1
        assert engine._cache_misses == 1

        engine.clear_cache()

        assert len(engine._session_cache) == 0
        assert engine._cache_hits == 0
        assert engine._cache_misses == 0

    def test_index_multiple_sessions(self):
        """测试批量索引多个会话."""
        engine = CachedSearchEngine()

        sessions = []
        for i in range(3):
            now = datetime.now()
            metadata = SessionMetadata(
                id=f"session-{i}",
                title=f"Session {i}",
                created_at=now,
                updated_at=now,
                model="gpt-4",
            )
            msg = UserMessage(content=f"Message with keyword {i}")
            session = ChatSession(metadata=metadata, messages=[msg])
            sessions.append(session)

        engine.index_sessions(sessions)

        assert len(engine._session_cache) == 3
        assert engine._cache_misses == 3

        # 搜索结果应包含所有会话
        result = engine.search("keyword")
        assert result.total_matches == 3

    def test_search_no_results(self):
        """测试无匹配结果的搜索."""
        engine = CachedSearchEngine()
        session = self._create_test_session()

        engine.index_session(session)
        result = engine.search("nonexistent")

        assert result.total_matches == 0
        assert result.query == "nonexistent"

    def test_search_case_insensitive(self):
        """测试不区分大小写搜索."""
        engine = CachedSearchEngine()
        session = self._create_test_session()

        engine.index_session(session)

        # 大写搜索应返回结果
        result_upper = engine.search("KEYWORD")
        result_lower = engine.search("keyword")

        assert result_upper.total_matches == result_lower.total_matches

    def test_search_case_sensitive(self):
        """测试区分大小写搜索."""
        engine = CachedSearchEngine()

        now = datetime.now()
        metadata = SessionMetadata(
            id="test-1",
            title="Test",
            created_at=now,
            updated_at=now,
            model="gpt-4",
        )
        msg = UserMessage(content="Hello KEYWORD")
        session = ChatSession(metadata=metadata, messages=[msg])

        engine.index_session(session)

        # 区分大小写搜索
        result_sensitive = engine.search("keyword", case_sensitive=True)
        result_insensitive = engine.search("keyword", case_sensitive=False)

        # 不区分大小写应找到结果，区分大小写找不到
        assert result_sensitive.total_matches == 0
        assert result_insensitive.total_matches == 1


class TestCachedSearchPerformance:
    """测试缓存搜索性能."""

    def _create_large_session(self, message_count: int) -> ChatSession:
        """创建包含大量消息的会话."""
        now = datetime.now()
        metadata = SessionMetadata(
            id="large-session",
            title="Large Session",
            created_at=now,
            updated_at=now,
            model="gpt-4",
            message_count=message_count,
        )

        messages = []
        for i in range(message_count):
            msg = UserMessage(
                content=f"This is message number {i} with some content text to search through",
                timestamp=now,
            )
            messages.append(msg)

        return ChatSession(metadata=metadata, messages=messages)

    def test_performance_with_cache(self):
        """测试缓存带来的性能提升."""
        import time

        engine = CachedSearchEngine()
        session = self._create_large_session(message_count=100)

        # 第一次索引和搜索
        start = time.time()
        engine.index_session(session)
        result1 = engine.search("message")
        first_time = time.time() - start

        assert result1.total_matches > 0

        # 重置索引（模拟重新索引相同会话）
        engine.clear_index()

        # 第二次（应使用缓存）
        start = time.time()
        engine.index_session(session)
        result2 = engine.search("message")
        second_time = time.time() - start

        assert result2.total_matches > 0

        # 缓存版本应显著更快（至少快 50%）
        assert second_time < first_time * 0.5 or second_time < 0.001

        # 验证缓存命中
        stats = engine.get_cache_stats()
        assert stats["hit_rate"] > 0

    def test_search_time_within_limit(self):
        """测试搜索时间是否在限制内（< 50ms）."""
        import time

        engine = CachedSearchEngine()
        session = self._create_large_session(message_count=500)

        engine.index_session(session)

        # 执行多次搜索
        times = []
        for _ in range(10):
            start = time.time()
            result = engine.search("content")
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)

        avg_time = sum(times) / len(times)

        # 平均搜索时间应小于 50ms
        assert avg_time < 50.0
        assert result.total_matches > 0


class TestCachedSearchEdgeCases:
    """测试缓存搜索边界情况."""

    def test_empty_session(self):
        """测试空会话."""
        engine = CachedSearchEngine()

        now = datetime.now()
        metadata = SessionMetadata(
            id="empty-session",
            title="Empty Session",
            created_at=now,
            updated_at=now,
            model="gpt-4",
            message_count=0,
        )
        session = ChatSession(metadata=metadata, messages=[])

        engine.index_session(session)

        result = engine.search("keyword")
        assert result.total_matches == 0

    def test_search_empty_query(self):
        """测试空查询."""
        engine = CachedSearchEngine()

        now = datetime.now()
        metadata = SessionMetadata(
            id="test-1",
            title="Test",
            created_at=now,
            updated_at=now,
            model="gpt-4",
        )
        msg = UserMessage(content="Hello world")
        session = ChatSession(metadata=metadata, messages=[msg])

        engine.index_session(session)

        # 空查询应匹配所有内容
        result = engine.search("")
        # 由于转义，空字符串会匹配任何位置
        assert isinstance(result, SearchResult)

    def test_search_with_regex(self):
        """测试正则表达式搜索."""
        engine = CachedSearchEngine()

        now = datetime.now()
        metadata = SessionMetadata(
            id="test-1",
            title="Test",
            created_at=now,
            updated_at=now,
            model="gpt-4",
        )
        msg = UserMessage(content="Hello 123 world")
        session = ChatSession(metadata=metadata, messages=[msg])

        engine.index_session(session)

        # 正则搜索
        result = engine.search(r"\d+", use_regex=True)
        assert result.total_matches == 1

    def test_invalid_regex_fallback(self):
        """测试无效正则回退."""
        engine = CachedSearchEngine()

        now = datetime.now()
        metadata = SessionMetadata(
            id="test-1",
            title="Test",
            created_at=now,
            updated_at=now,
            model="gpt-4",
        )
        msg = UserMessage(content="Hello (world")
        session = ChatSession(metadata=metadata, messages=[msg])

        engine.index_session(session)

        # 无效正则应作为普通字符串搜索
        result = engine.search("(world", use_regex=True)
        assert result.total_matches == 1

    def test_max_results_limit(self):
        """测试最大结果数限制."""
        engine = CachedSearchEngine(max_results=5)

        now = datetime.now()
        metadata = SessionMetadata(
            id="test-1",
            title="Test",
            created_at=now,
            updated_at=now,
            model="gpt-4",
        )

        # 创建10条包含相同关键词的消息
        messages = []
        for i in range(10):
            msg = UserMessage(content=f"Message {i} with keyword")
            messages.append(msg)

        session = ChatSession(metadata=metadata, messages=messages)
        engine.index_session(session)

        result = engine.search("keyword")
        assert result.total_matches == 5  # 受 max_results 限制

    def test_assistant_message_with_empty_content(self):
        """测试助手消息空内容处理."""
        engine = CachedSearchEngine()

        now = datetime.now()
        metadata = SessionMetadata(
            id="test-1",
            title="Test",
            created_at=now,
            updated_at=now,
            model="gpt-4",
        )

        # 创建包含空内容的助手消息
        msg = AssistantMessage(content=[], timestamp=now)
        session = ChatSession(metadata=metadata, messages=[msg])

        engine.index_session(session)

        # 应正常处理，不抛出异常
        result = engine.search("keyword")
        assert result.total_matches == 0
