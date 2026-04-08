"""消息搜索模块 - 支持会话历史搜索."""

from .engine import (
    MessageSearchEngine,
    SearchScope,
    SearchResult,
    SearchMatch,
)
from .cache import CachedSearchEngine

__all__ = [
    "MessageSearchEngine",
    "CachedSearchEngine",
    "SearchScope",
    "SearchResult",
    "SearchMatch",
]
