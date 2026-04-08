"""会话管理器."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from .models import ChatSession, SessionMetadata
from .storage import SessionStorage


class SessionManager:
    """会话管理器."""

    def __init__(self, storage: SessionStorage):
        self._storage = storage
        self._current_session: Optional[ChatSession] = None

    def create(self, title: str, model: str) -> ChatSession:
        """创建新会话."""
        metadata = SessionMetadata(
            id=str(uuid.uuid4())[:8],
            title=title,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            model=model,
        )
        session = ChatSession(metadata=metadata)
        self._storage.save(session)
        self._current_session = session
        return session

    def load(self, session_id: str) -> Optional[ChatSession]:
        """加载并切换到指定会话."""
        session = self._storage.load(session_id)
        if session:
            self._current_session = session
        return session

    def current(self) -> Optional[ChatSession]:
        """获取当前会话."""
        return self._current_session

    def list_all(self) -> List[ChatSession]:
        """列出所有会话."""
        return self._storage.list_all()

    def delete(self, session_id: str) -> bool:
        """删除会话."""
        file_path = self._storage._storage_dir / f"{session_id}.json"
        if file_path.exists():
            file_path.unlink()
            if (
                self._current_session
                and self._current_session.metadata.id == session_id
            ):
                self._current_session = None
            return True
        return False
