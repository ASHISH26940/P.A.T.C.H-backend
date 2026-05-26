from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.services.memory_service import MemoryService
from app.core.database import get_db
from loguru import logger

router = APIRouter()


async def get_memory_service(db: AsyncSession = Depends(get_db)) -> MemoryService:
    return MemoryService(db=db)


async def get_chat_service(
    memory_service: MemoryService = Depends(get_memory_service),
) -> ChatService:
    return ChatService(memory_service=memory_service)


@router.post("/", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_with_ai(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    logger.info(f"Chat request from user '{request.user_id}'")
    try:
        return await chat_service.process_chat_message(request)
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail="Chat processing failed")
