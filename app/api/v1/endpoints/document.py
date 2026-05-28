from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from loguru import logger

from app.core.database import get_db, User
from app.core.security import get_current_user
from app.services.memory_service import MemoryService
from pydantic import BaseModel, Field

router = APIRouter()


class MemoryCreate(BaseModel):
    content: str = Field(..., description="Memory content")
    memory_type: str = Field("general", description="Type of memory")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Importance score")


class MemoryResponse(BaseModel):
    id: str
    user_id: str
    memory_type: str
    content: str
    importance: float
    created_at: str

    class Config:
        from_attributes = True


class MemorySearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    n_results: int = Field(5, ge=1, le=50)
    memory_type: Optional[str] = None


class MemorySearchResponse(BaseModel):
    results: List[MemoryResponse]


async def get_memory_service(db: AsyncSession = Depends(get_db)) -> MemoryService:
    return MemoryService(db=db)


@router.post("/memories", response_model=MemoryResponse, status_code=201)
async def create_memory(
    memory_data: MemoryCreate,
    current_user: User = Depends(get_current_user),
    service: MemoryService = Depends(get_memory_service),
):
    memory = await service.add_memory(
        user_id=current_user.username,
        content=memory_data.content,
        memory_type=memory_data.memory_type,
        importance=memory_data.importance,
    )
    return MemoryResponse(
        id=memory.id,
        user_id=memory.user_id,
        memory_type=memory.memory_type,
        content=memory.content,
        importance=memory.importance,
        created_at=memory.created_at.isoformat(),
    )


@router.post("/memories/search", response_model=MemorySearchResponse)
async def search_memories(
    search: MemorySearchRequest,
    current_user: User = Depends(get_current_user),
    service: MemoryService = Depends(get_memory_service),
):
    results = await service.search_memories(
        user_id=current_user.username,
        query=search.query,
        n_results=search.n_results,
        memory_type=search.memory_type,
    )
    return MemorySearchResponse(
        results=[
            MemoryResponse(
                id=m.id,
                user_id=m.user_id,
                memory_type=m.memory_type,
                content=m.content,
                importance=m.importance,
                created_at=m.created_at.isoformat() if m.created_at else "",
            )
            for m in results
        ]
    )


@router.get("/memories/recent", response_model=List[MemoryResponse])
async def get_recent_memories(
    limit: int = 10,
    memory_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    service: MemoryService = Depends(get_memory_service),
):
    results = await service.get_recent_memories(
        user_id=current_user.username, limit=limit, memory_type=memory_type
    )
    return [
        MemoryResponse(
            id=m.id,
            user_id=m.user_id,
            memory_type=m.memory_type,
            content=m.content,
            importance=m.importance,
            created_at=m.created_at.isoformat() if m.created_at else "",
        )
        for m in results
    ]


@router.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    service: MemoryService = Depends(get_memory_service),
):
    memory = await service.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryResponse(
        id=memory.id,
        user_id=memory.user_id,
        memory_type=memory.memory_type,
        content=memory.content,
        importance=memory.importance,
        created_at=memory.created_at.isoformat() if memory.created_at else "",
    )


class MemoryUpdate(BaseModel):
    content: Optional[str] = None
    memory_type: Optional[str] = None
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)


@router.put("/memories/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    update_data: MemoryUpdate,
    service: MemoryService = Depends(get_memory_service),
):
    memory = await service.update_memory(
        memory_id=memory_id,
        content=update_data.content,
        memory_type=update_data.memory_type,
        importance=update_data.importance,
    )
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryResponse(
        id=memory.id,
        user_id=memory.user_id,
        memory_type=memory.memory_type,
        content=memory.content,
        importance=memory.importance,
        created_at=memory.created_at.isoformat() if memory.created_at else "",
    )


@router.delete("/memories/{memory_id}", status_code=204)
async def delete_memory(
    memory_id: str,
    service: MemoryService = Depends(get_memory_service),
):
    deleted = await service.delete_memory(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return
