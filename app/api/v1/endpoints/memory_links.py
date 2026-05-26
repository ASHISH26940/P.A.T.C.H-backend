from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.core.database import get_db
from app.services.memory_service import MemoryService
from app.schemas.memory_link import MemoryLinkCreate, MemoryLinkResponse, MemoryLinkListResponse

router = APIRouter()


async def get_memory_service(db: AsyncSession = Depends(get_db)) -> MemoryService:
    return MemoryService(db=db)


@router.post("/links", response_model=MemoryLinkResponse, status_code=201)
async def create_memory_link(
    link_data: MemoryLinkCreate,
    service: MemoryService = Depends(get_memory_service),
):
    source = await service.get_memory(link_data.source_memory_id)
    target = await service.get_memory(link_data.target_memory_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source memory {link_data.source_memory_id} not found")
    if not target:
        raise HTTPException(status_code=404, detail=f"Target memory {link_data.target_memory_id} not found")

    link = await service.link_memories(
        source_memory_id=link_data.source_memory_id,
        target_memory_id=link_data.target_memory_id,
        relationship=link_data.relationship,
    )
    return MemoryLinkResponse(
        id=link.id,
        source_memory_id=link.source_memory_id,
        target_memory_id=link.target_memory_id,
        relationship=link.relationship,
        created_at=link.created_at.isoformat(),
    )


@router.get("/links/{memory_id}", response_model=MemoryLinkListResponse)
async def get_memory_links_endpoint(
    memory_id: str,
    relationship: Optional[str] = None,
    service: MemoryService = Depends(get_memory_service),
):
    memory = await service.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    links = await service.get_memory_links(memory_id, relationship=relationship)
    return MemoryLinkListResponse(
        links=[
            MemoryLinkResponse(
                id=l.id,
                source_memory_id=l.source_memory_id,
                target_memory_id=l.target_memory_id,
                relationship=l.relationship,
                created_at=l.created_at.isoformat(),
            )
            for l in links
        ]
    )


@router.delete("/links/{link_id}", status_code=204)
async def delete_memory_link(
    link_id: str,
    service: MemoryService = Depends(get_memory_service),
):
    deleted = await service.delete_link(link_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Link not found")
    return
