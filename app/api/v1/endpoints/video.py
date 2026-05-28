from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.database import get_db, User
from app.core.security import get_current_user
from app.services.video_service import VideoService, VideoIngestionError
from app.schemas.video import VideoIngestRequest, VideoIngestResponse, VideoMemoryResult

router = APIRouter()


@router.post("/ingest", response_model=VideoIngestResponse)
async def ingest_video(
    request: VideoIngestRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = VideoService(db=db)
    try:
        result = await service.ingest(current_user.username, request.url, current_user.youtube_cookies)
    except VideoIngestionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return VideoIngestResponse(
        video_title=result["video_title"],
        channel=result["channel"],
        duration=result["duration"],
        subtitles_available=result["subtitles_available"],
        memories=[VideoMemoryResult(**m) for m in result["memories"]],
    )
