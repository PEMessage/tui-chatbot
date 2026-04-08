"""会话数据模型."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from ..agent.types import AgentMessage


class SessionMetadata(BaseModel):
    """会话元数据."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    model: str
    message_count: int = 0


class ChatSession(BaseModel):
    """聊天会话."""

    metadata: SessionMetadata
    messages: List[AgentMessage] = Field(default_factory=list)

    def add_message(self, message: AgentMessage) -> None:
        """添加消息并更新统计."""
        self.messages.append(message)
        self.metadata.message_count = len(self.messages)
        self.metadata.updated_at = datetime.now()
