"""会话存储管理."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .models import ChatSession


class SessionStorage:
    """会话存储管理."""

    def __init__(self, storage_dir: Optional[Path] = None):
        if storage_dir is None:
            storage_dir = Path.home() / ".config" / "tui-chatbot" / "sessions"
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, session: ChatSession) -> None:
        """保存会话到文件."""
        file_path = self._storage_dir / f"{session.metadata.id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session.model_dump(), f, indent=2, default=str)

    def load(self, session_id: str) -> Optional[ChatSession]:
        """加载会话."""
        file_path = self._storage_dir / f"{session_id}.json"
        if not file_path.exists():
            return None
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return ChatSession.model_validate(data)

    def list_all(self) -> List[ChatSession]:
        """列出所有会话."""
        sessions = []
        for file_path in self._storage_dir.glob("*.json"):
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            sessions.append(ChatSession.model_validate(data))
        return sorted(sessions, key=lambda s: s.metadata.updated_at, reverse=True)
