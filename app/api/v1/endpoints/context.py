from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.schemas.context import DynamicContext, ContextResponse
from app.core.database import get_db
from app.services.context_injector import ContextInjectorService

router = APIRouter()


async def get_context_service(db: AsyncSession = Depends(get_db)) -> ContextInjectorService:
    return ContextInjectorService(db=db)


@router.put("/{user_id}", response_model=ContextResponse)
async def update_user_context(
    user_id: str,
    context_data: DynamicContext,
    service: ContextInjectorService = Depends(get_context_service),
):
    logger.info(f"Updating context for user {user_id}")
    updated = await service.update_context(user_id, context_data.context_data)
    return ContextResponse(
        user_id=user_id,
        context_data={k: v for k, v in updated.items() if k != "updated_at"},
        updated_at=updated.get("updated_at", "N/A"),
    )


@router.get("/{user_id}", response_model=ContextResponse)
async def get_user_context(
    user_id: str,
    service: ContextInjectorService = Depends(get_context_service),
):
    logger.info(f"Getting context for user {user_id}")
    ctx = await service.get_current_context(user_id)
    if not ctx:
        return ContextResponse(user_id=user_id, context_data={}, updated_at="N/A")
    updated_at = ctx.pop("updated_at", "N/A")
    return ContextResponse(user_id=user_id, context_data=ctx, updated_at=updated_at)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_context(
    user_id: str,
    service: ContextInjectorService = Depends(get_context_service),
):
    logger.info(f"Deleting context for user {user_id}")
    deleted = await service.delete_context(user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Context not found")
    return
