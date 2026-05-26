from pydantic import BaseModel, Field


class VideoIngestRequest(BaseModel):
    url: str = Field(..., description="YouTube or video URL to ingest")


class VideoMemoryResult(BaseModel):
    id: str
    content: str
    memory_type: str
    importance: float


class VideoIngestResponse(BaseModel):
    video_title: str
    channel: str | None = None
    duration: int | None = None
    subtitles_available: bool = False
    memories: list[VideoMemoryResult]
