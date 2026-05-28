import uuid
import datetime
import math
import numpy as np
from typing import List, Optional
from sqlalchemy import select, desc, or_
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.models.memory import Memory, ChatMessage, Extraction, MemoryLink, UserContext
from app.core.llm_client import get_embeddings_model
from app.core.config import settings


class MemoryService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.embedder = get_embeddings_model()

    async def add_memory(
        self,
        user_id: str,
        content: str,
        memory_type: str = "general",
        metadata: Optional[dict] = None,
        importance: float = 0.5,
    ) -> Memory:
        existing = await self.db.execute(
            select(Memory).where(Memory.user_id == user_id, Memory.content == content).limit(1)
        )
        dup = existing.scalar_one_or_none()
        if dup:
            logger.info(f"Skipped duplicate memory for user {user_id} — content already exists as {dup.id}")
            return dup

        embedding = None
        try:
            embedding = self.embedder.embed_documents([content])[0]
        except Exception as e:
            logger.warning(f"Embedding failed for memory (storing without vector): {e}")
        memory = Memory(
            id=str(uuid.uuid4()),
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            metadata_=metadata or {},
            importance=importance,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )
        self.db.add(memory)
        await self.db.commit()
        await self.db.refresh(memory)
        logger.info(f"Stored memory {memory.id} (type={memory_type}) for user {user_id}")
        return memory

    async def search_memories(
        self,
        user_id: str,
        query: str,
        n_results: int = 5,
        memory_type: Optional[str] = None,
        min_similarity: float = 0.3,
    ) -> List[Memory]:
        try:
            query_vec = np.array(self.embedder.embed_query(query))
        except Exception as e:
            logger.warning(f"Embedding failed for search query (returning empty): {e}")
            return []

        filters = [Memory.user_id == user_id]
        if memory_type:
            filters.append(Memory.memory_type == memory_type)

        stmt = (
            select(Memory)
            .where(*filters)
            .limit(n_results * 4)
        )
        result = await self.db.execute(stmt)
        candidates = result.scalars().all()

        scored = []
        for m in candidates:
            if m.embedding is None:
                continue
            mem_vec = np.array(m.embedding)
            cos_sim = float(np.dot(mem_vec, query_vec) / (np.linalg.norm(mem_vec) * np.linalg.norm(query_vec) + 1e-10))
            if cos_sim < min_similarity:
                continue
            last_access = m.last_accessed_at
            if last_access is not None and last_access.tzinfo is None:
                last_access = last_access.replace(tzinfo=datetime.timezone.utc)
            days_since_access = (
                (datetime.datetime.now(datetime.timezone.utc) - last_access).days
                if last_access is not None
                else settings.MEMORY_IMPORTANCE_DECAY_DAYS
            )
            recency_factor = math.exp(-days_since_access / settings.MEMORY_IMPORTANCE_DECAY_DAYS)
            score = cos_sim * 0.6 + recency_factor * 0.2 + m.importance * 0.2
            scored.append((score, m))

            m.last_accessed_at = datetime.datetime.now(datetime.timezone.utc)
            m.access_count = (m.access_count or 0) + 1

        scored.sort(key=lambda x: x[0], reverse=True)
        await self.db.commit()

        return [m for _, m in scored[:n_results]]

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        try:
            return await self.db.get(Memory, memory_id)
        except Exception:
            return None

    async def update_memory(
        self, memory_id: str, content: Optional[str] = None,
        memory_type: Optional[str] = None, importance: Optional[float] = None,
    ) -> Optional[Memory]:
        memory = await self.db.get(Memory, memory_id)
        if not memory:
            return None
        if content is not None:
            memory.content = content
        if memory_type is not None:
            memory.memory_type = memory_type
        if importance is not None:
            memory.importance = importance
        await self.db.commit()
        await self.db.refresh(memory)
        return memory

    async def delete_memory(self, memory_id: str) -> bool:
        memory = await self.db.get(Memory, memory_id)
        if not memory:
            return False
        await self.db.delete(memory)
        await self.db.commit()
        return True

    async def get_recent_memories(
        self, user_id: str, limit: int = 10, memory_type: Optional[str] = None
    ) -> List[Memory]:
        filters = [Memory.user_id == user_id]
        if memory_type:
            filters.append(Memory.memory_type == memory_type)
        stmt = (
            select(Memory)
            .where(*filters)
            .order_by(desc(Memory.created_at))
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        items = list(result.scalars().all())
        seen = set()
        deduped = []
        for m in items:
            if m.content in seen: continue
            seen.add(m.content)
            deduped.append(m)
        return deduped

    async def get_important_memories(
        self, user_id: str, limit: int = 5
    ) -> List[Memory]:
        stmt = (
            select(Memory)
            .where(Memory.user_id == user_id)
            .order_by(desc(Memory.importance), desc(Memory.access_count))
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def add_chat_message(
        self,
        user_id: str,
        role: str,
        content: str,
        session_id: Optional[str] = None,
    ) -> ChatMessage:
        msg = ChatMessage(
            id=str(uuid.uuid4()),
            user_id=user_id,
            role=role,
            content=content,
            session_id=session_id,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )
        self.db.add(msg)
        await self.db.commit()
        await self.db.refresh(msg)
        return msg

    async def get_chat_history(
        self,
        user_id: str,
        limit: Optional[int] = None,
        before_id: Optional[str] = None,
    ) -> List[ChatMessage]:
        filters = [ChatMessage.user_id == user_id]
        if before_id:
            before_msg = await self.db.get(ChatMessage, before_id)
            if before_msg:
                filters.append(ChatMessage.created_at < before_msg.created_at)
        stmt = (
            select(ChatMessage)
            .where(*filters)
            .order_by(desc(ChatMessage.created_at))
        )
        if limit:
            stmt = stmt.limit(limit)
        result = await self.db.execute(stmt)
        msgs = list(result.scalars().all())
        msgs.reverse()
        return msgs

    async def clear_chat_history(self, user_id: str) -> int:
        stmt = select(ChatMessage).where(ChatMessage.user_id == user_id)
        result = await self.db.execute(stmt)
        msgs = result.scalars().all()
        count = len(msgs)
        for m in msgs:
            await self.db.delete(m)
        await self.db.commit()
        return count

    async def add_extraction(
        self,
        user_id: str,
        insights: dict,
        session_id: Optional[str] = None,
        source: str = "chat",
    ) -> Extraction:
        extraction = Extraction(
            id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            source=source,
            extracted_insights=insights,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )
        self.db.add(extraction)
        await self.db.commit()
        await self.db.refresh(extraction)
        logger.info(f"Stored extraction {extraction.id} (source={source}) for user {user_id}")
        return extraction

    async def delete_link(self, link_id: str) -> bool:
        try:
            link = await self.db.get(MemoryLink, link_id)
        except Exception:
            return False
        if not link:
            return False
        await self.db.delete(link)
        await self.db.commit()
        return True

    async def link_memories(
        self,
        source_memory_id: str,
        target_memory_id: str,
        relationship: str = "related_to",
    ) -> MemoryLink:
        link = MemoryLink(
            id=str(uuid.uuid4()),
            source_memory_id=source_memory_id,
            target_memory_id=target_memory_id,
            relationship=relationship,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )
        self.db.add(link)
        await self.db.commit()
        await self.db.refresh(link)
        return link

    async def get_context(self, user_id: str) -> Optional[dict]:
        ctx = await self.db.get(UserContext, user_id)
        if not ctx:
            return None
        data = dict(ctx.context_data)
        data["updated_at"] = ctx.updated_at.isoformat() if ctx.updated_at else None
        return data

    async def update_context(self, user_id: str, context_data: dict) -> dict:
        ctx = await self.db.get(UserContext, user_id)
        if ctx:
            ctx.context_data.update(context_data)
            ctx.updated_at = datetime.datetime.now(datetime.timezone.utc)
        else:
            ctx = UserContext(
                user_id=user_id,
                context_data=context_data,
                updated_at=datetime.datetime.now(datetime.timezone.utc),
            )
            self.db.add(ctx)
        await self.db.commit()
        await self.db.refresh(ctx)
        result = dict(ctx.context_data)
        result["updated_at"] = ctx.updated_at.isoformat()
        return result

    async def delete_context(self, user_id: str) -> bool:
        ctx = await self.db.get(UserContext, user_id)
        if not ctx:
            return False
        await self.db.delete(ctx)
        await self.db.commit()
        return True

    async def get_all_links(self) -> List[MemoryLink]:
        result = await self.db.execute(select(MemoryLink))
        return list(result.scalars().all())

    async def get_memory_links(
        self, memory_id: str, relationship: Optional[str] = None
    ) -> List[MemoryLink]:
        filters = or_(
            MemoryLink.source_memory_id == memory_id,
            MemoryLink.target_memory_id == memory_id,
        )
        stmt = select(MemoryLink).where(filters)
        if relationship:
            stmt = stmt.where(MemoryLink.relationship == relationship)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_linked_memories(
        self, memory_id: str, relationship: Optional[str] = None
    ) -> List[Memory]:
        links = await self.get_memory_links(memory_id, relationship=relationship)

        linked_ids = set()
        for link in links:
            if link.source_memory_id != memory_id:
                linked_ids.add(link.source_memory_id)
            if link.target_memory_id != memory_id:
                linked_ids.add(link.target_memory_id)

        if not linked_ids:
            return []

        memories_result = await self.db.execute(
            select(Memory).where(Memory.id.in_(linked_ids))
        )
        return list(memories_result.scalars().all())
