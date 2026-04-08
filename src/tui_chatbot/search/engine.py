"""消息搜索引擎."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import List, Optional, Union

from ..agent.types import AgentMessage, UserMessage, AssistantMessage, TextContent
from ..session.models import ChatSession


class SearchScope(Enum):
    """搜索范围."""

    ALL = auto()  # 所有消息
    USER_ONLY = auto()  # 仅用户消息
    ASSISTANT_ONLY = auto()  # 仅助手消息


@dataclass
class SearchMatch:
    """单个匹配结果."""

    message: AgentMessage
    session_id: str
    session_title: str
    matched_text: str  # 匹配的文本片段
    match_start: int  # 匹配起始位置
    match_end: int  # 匹配结束位置
    context_before: str  # 前文上下文
    context_after: str  # 后文上下文
    timestamp: datetime


@dataclass
class SearchResult:
    """搜索结果集合."""

    query: str
    total_matches: int
    matches: List[SearchMatch]
    scope: SearchScope
    search_time_ms: float


class MessageSearchEngine:
    """消息搜索引擎 - 支持会话历史搜索.

    特性:
        - 支持关键词搜索（不区分大小写）
        - 支持正则表达式搜索（可选）
        - 支持按角色过滤
        - 返回带上下文的高亮结果

    Usage:
        >>> engine = MessageSearchEngine()
        >>> engine.index_session(session)
        >>> result = engine.search("关键词")
        >>> for match in result.matches:
        ...     print(f"{match.session_title}: {match.matched_text}")
    """

    def __init__(
        self,
        context_chars: int = 30,  # 上下文字符数
        max_results: int = 100,  # 最大结果数
    ):
        self._context_chars = context_chars
        self._max_results = max_results
        self._index: List[tuple[ChatSession, AgentMessage]] = []

    def index_session(self, session: ChatSession) -> None:
        """索引会话消息.

        Args:
            session: 要索引的会话
        """
        for msg in session.messages:
            self._index.append((session, msg))

    def index_sessions(self, sessions: List[ChatSession]) -> None:
        """批量索引多个会话."""
        for session in sessions:
            self.index_session(session)

    def clear_index(self) -> None:
        """清空索引."""
        self._index.clear()

    def search(
        self,
        query: str,
        scope: SearchScope = SearchScope.ALL,
        use_regex: bool = False,
        case_sensitive: bool = False,
    ) -> SearchResult:
        """搜索消息.

        Args:
            query: 搜索关键词或正则表达式
            scope: 搜索范围
            use_regex: 是否使用正则表达式
            case_sensitive: 是否区分大小写

        Returns:
            SearchResult: 搜索结果
        """
        start_time = time.time()
        matches: List[SearchMatch] = []

        # 编译搜索模式
        flags = 0 if case_sensitive else re.IGNORECASE
        if use_regex:
            try:
                pattern = re.compile(query, flags)
            except re.error:
                # 无效正则，当作普通字符串
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
        if isinstance(message, UserMessage):
            return message.content
        elif isinstance(message, AssistantMessage):
            # 合并所有文本内容
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
    ) -> SearchMatch:
        """创建匹配对象."""
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
