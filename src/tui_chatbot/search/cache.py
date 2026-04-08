"""搜索索引缓存模块."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any

from .engine import SearchScope, SearchResult
from ..session.models import ChatSession
from ..agent.types import AgentMessage


@dataclass
class CacheEntry:
    """缓存条目."""

    messages: List[AgentMessage]  # 缓存的消息列表
    session_updated_at: datetime
    created_at: datetime


class CachedSearchEngine:
    """带缓存的消息搜索引擎.

    特性:
        - 缓存会话索引，避免重复构建
        - 根据 session.updated_at 自动失效缓存
        - 支持缓存统计和手动清理

    Usage:
        >>> engine = CachedSearchEngine()
        >>> engine.index_session(session)
        >>> result = engine.search("keyword")
        >>> stats = engine.get_cache_stats()
    """

    def __init__(
        self,
        context_chars: int = 30,
        max_results: int = 100,
        max_cache_size: int = 100,
    ):
        self._context_chars = context_chars
        self._max_results = max_results
        self._max_cache_size = max_cache_size

        # 缓存: session_id -> CacheEntry
        self._session_cache: Dict[str, CacheEntry] = {}

        # 当前索引: 列表存储 (session, message) 元组
        self._index: List[tuple[ChatSession, AgentMessage]] = []

        # 缓存统计
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_cache_key(self, session: ChatSession) -> str:
        """生成缓存键（基于会话ID和更新时间）."""
        return f"{session.metadata.id}:{session.metadata.updated_at.isoformat()}"

    def _build_index_for_session(self, session: ChatSession) -> List[AgentMessage]:
        """为会话构建消息索引.

        Returns:
            消息列表（用于缓存和索引）
        """
        return list(session.messages)

    def _cache_session(
        self, session: ChatSession, messages: List[AgentMessage]
    ) -> None:
        """缓存会话消息."""
        cache_key = self._get_cache_key(session)

        # 缓存大小管理
        if len(self._session_cache) >= self._max_cache_size:
            # 移除最旧的条目
            oldest_key = min(
                self._session_cache.items(),
                key=lambda x: x[1].created_at,
            )[0]
            del self._session_cache[oldest_key]

        self._session_cache[cache_key] = CacheEntry(
            messages=messages,
            session_updated_at=session.metadata.updated_at,
            created_at=datetime.now(),
        )

    def index_session(self, session: ChatSession, force: bool = False) -> None:
        """索引会话（支持缓存）.

        Args:
            session: 要索引的会话
            force: 强制重新索引，忽略缓存
        """
        cache_key = self._get_cache_key(session)

        # 检查缓存是否有效
        if not force and cache_key in self._session_cache:
            cached_entry = self._session_cache[cache_key]
            if cached_entry.session_updated_at == session.metadata.updated_at:
                # 缓存命中，直接使用缓存的消息
                self._cache_hits += 1
                for msg in cached_entry.messages:
                    self._index.append((session, msg))
                return

        # 缓存未命中，重新构建索引
        self._cache_misses += 1
        messages = self._build_index_for_session(session)

        # 缓存会话
        self._cache_session(session, messages)

        # 添加到索引
        for msg in messages:
            self._index.append((session, msg))

    def index_sessions(self, sessions: List[ChatSession], force: bool = False) -> None:
        """批量索引多个会话."""
        for session in sessions:
            self.index_session(session, force=force)

    def clear_index(self) -> None:
        """清空索引（保留缓存）."""
        self._index.clear()

    def clear_cache(self) -> None:
        """清空缓存."""
        self._session_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def search(
        self,
        query: str,
        scope: SearchScope = SearchScope.ALL,
        use_regex: bool = False,
        case_sensitive: bool = False,
    ) -> SearchResult:
        """搜索消息.

        Args:
            query: 搜索关键词
            scope: 搜索范围
            use_regex: 是否使用正则表达式
            case_sensitive: 是否区分大小写

        Returns:
            SearchResult: 搜索结果
        """
        import re
        import time

        from ..agent.types import UserMessage, AssistantMessage, TextContent

        start_time = time.time()
        matches = []

        # 编译搜索模式
        flags = 0 if case_sensitive else re.IGNORECASE
        if use_regex:
            try:
                pattern = re.compile(query, flags)
            except re.error:
                pattern = re.compile(re.escape(query), flags)
        else:
            pattern = re.compile(re.escape(query), flags)

        # 执行搜索
        for session, message in self._index:
            # 范围过滤
            if scope == SearchScope.USER_ONLY and not isinstance(message, UserMessage):
                continue
            if scope == SearchScope.ASSISTANT_ONLY and not isinstance(
                message, AssistantMessage
            ):
                continue

            # 获取消息内容
            content = self._get_message_content(message)
            if not content:
                continue

            # 搜索匹配
            for match in pattern.finditer(content):
                if len(matches) >= self._max_results:
                    break

                match_obj = self._create_match(
                    session, message, content, match.start(), match.end()
                )
                matches.append(match_obj)

            if len(matches) >= self._max_results:
                break

        search_time_ms = (time.time() - start_time) * 1000

        from .engine import SearchResult

        return SearchResult(
            query=query,
            total_matches=len(matches),
            matches=matches,
            scope=scope,
            search_time_ms=search_time_ms,
        )

    def search_current_session(
        self,
        session: ChatSession,
        query: str,
        scope: SearchScope = SearchScope.ALL,
        use_regex: bool = False,
        case_sensitive: bool = False,
    ) -> SearchResult:
        """在当前会话中搜索.

        Args:
            session: 当前会话
            query: 搜索关键词
            scope: 搜索范围
            use_regex: 是否使用正则表达式
            case_sensitive: 是否区分大小写

        Returns:
            SearchResult: 搜索结果
        """
        # 临时创建新索引，仅包含当前会话
        original_index = self._index.copy()
        self._index = []
        self.index_session(session)

        try:
            result = self.search(query, scope, use_regex, case_sensitive)
        finally:
            # 恢复原始索引
            self._index = original_index

        return result

    def _get_message_content(self, message: AgentMessage) -> str:
        """提取消息文本内容."""
        from ..agent.types import UserMessage, AssistantMessage, TextContent

        if isinstance(message, UserMessage):
            return message.content
        elif isinstance(message, AssistantMessage):
            texts = []
            for content in message.content:
                if isinstance(content, TextContent):
                    texts.append(content.text)
            return " ".join(texts)
        return ""

    def _create_match(
        self,
        session: ChatSession,
        message: AgentMessage,
        content: str,
        start: int,
        end: int,
    ):
        """创建匹配对象."""
        from .engine import SearchMatch

        # 计算上下文
        context_start = max(0, start - self._context_chars)
        context_end = min(len(content), end + self._context_chars)

        # 获取消息时间戳
        if hasattr(message, "timestamp"):
            timestamp = message.timestamp
        else:
            timestamp = session.metadata.updated_at

        return SearchMatch(
            message=message,
            session_id=session.metadata.id,
            session_title=session.metadata.title,
            matched_text=content[start:end],
            match_start=start,
            match_end=end,
            context_before=content[context_start:start],
            context_after=content[end:context_end],
            timestamp=timestamp,
        )

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息.

        Returns:
            Dict 包含:
                - hits: 缓存命中次数
                - misses: 缓存未命中次数
                - hit_rate: 命中率 (0-1)
                - cached_sessions: 缓存的会话数量
                - max_cache_size: 最大缓存大小
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0

        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cached_sessions": len(self._session_cache),
            "max_cache_size": self._max_cache_size,
        }
