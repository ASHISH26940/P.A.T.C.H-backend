from pydantic import BaseModel, Field
from typing import List, Optional


class MemoryLinkCreate(BaseModel):
    source_memory_id: str
    target_memory_id: str
    relationship: str = Field("related_to", description="Type of relationship (e.g. related_to, depends_on, contradicts)")


class MemoryLinkResponse(BaseModel):
    id: str
    source_memory_id: str
    target_memory_id: str
    relationship: str
    created_at: str

    class Config:
        from_attributes = True


class MemoryLinkListResponse(BaseModel):
    links: List[MemoryLinkResponse]
