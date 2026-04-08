"""消息搜索模块 - 支持会话历史搜索."""

from .engine import (
    MessageSearchEngine,
    SearchScope,
    SearchResult,
    SearchMatch,
)

__all__ = [
    "MessageSearchEngine",
    "SearchScope",
    "SearchResult",
    "SearchMatch",
]
