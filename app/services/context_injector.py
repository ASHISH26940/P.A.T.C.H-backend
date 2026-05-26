from typing import Dict, Any, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.services.memory_service import MemoryService


class ContextInjectorService:
    def __init__(self, db: AsyncSession):
        self.memory = MemoryService(db)

    async def get_current_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        return await self.memory.get_context(user_id)

    async def update_context(
        self, user_id: str, new_context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        return await self.memory.update_context(user_id, new_context_data)

    async def delete_context(self, user_id: str) -> bool:
        return await self.memory.delete_context(user_id)

    async def add_message_to_history(self, user_id: str, role: str, content: str):
        await self.memory.add_chat_message(user_id=user_id, role=role, content=content)

    async def get_chat_history(
        self, user_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        messages = await self.memory.get_chat_history(user_id=user_id, limit=limit)
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.created_at.isoformat() if msg.created_at else None,
            }
            for msg in messages
        ]
