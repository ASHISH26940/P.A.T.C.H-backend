from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from app.models.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.memory_service import MemoryService
from app.core.database import get_db
from app.core.security import get_current_user
from app.core.database import User as DBUser
from app.models.memory import ChatMessage as DBChatMessage
from loguru import logger

router = APIRouter()


async def get_memory_service(db: AsyncSession = Depends(get_db)) -> MemoryService:
    return MemoryService(db=db)


async def get_chat_service(
    memory_service: MemoryService = Depends(get_memory_service),
) -> ChatService:
    return ChatService(memory_service=memory_service)


class ChatHistoryMessage(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime
    session_id: Optional[str] = None

class ChatHistoryResponse(BaseModel):
    messages: List[ChatHistoryMessage]
    total: int

@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: DBUser = Depends(get_current_user),
):
    result = await db.execute(
        select(DBChatMessage)
        .where(DBChatMessage.user_id == str(current_user.id))
        .order_by(desc(DBChatMessage.created_at))
        .limit(limit)
    )
    msgs = list(result.scalars().all())
    msgs.reverse()
    return ChatHistoryResponse(
        messages=[
            ChatHistoryMessage(
                id=m.id, role="human" if m.role == "human" else "model",
                content=m.content, created_at=m.created_at, session_id=m.session_id
            )
            for m in msgs
        ],
        total=len(msgs),
    )

@router.post("/", status_code=status.HTTP_200_OK)
async def chat_with_ai(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    logger.info(f"Chat request from user '{request.user_id}'")
    return StreamingResponse(
        chat_service.process_chat_message_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
