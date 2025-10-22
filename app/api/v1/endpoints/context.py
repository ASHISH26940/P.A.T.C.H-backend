from fastapi import APIRouter, Depends, HTTPException, status
import redis.asyncio as redis
from loguru import logger

from app.core.redis_client import get_redis_client
from app.schemas.context import DynamicContext, ContextResponse
from app.core.database import get_db # Imported but not used in these endpoints
from app.models.context import Context as ContextModel # Imported but not used in these endpoints
from app.services.context_injector import ContextInjectorService
from app.models.common import UserID

router = APIRouter() # No prefix here. If you want /context, specify it like router = APIRouter(prefix="/context", tags=["Context"])


@router.put("/{user_id}", response_model=ContextResponse) # Changed {user_Id} to {user_id} for consistency
async def update_user_context(
    user_id: UserID,
    context_data: DynamicContext,
    redis_client: redis.Redis = Depends(get_redis_client),
):
    """
    Update or merge dynamic context data for a specific user.
    Existing keys will be overwritten; new keys will be added.
    """
    logger.info(f"Attempting to update context for user_id:{user_id}")
    service =ContextInjectorService(redis_client=redis_client)
    
    # updated_context_flat will be the flat dictionary from Redis, e.g., {'key': 'value', 'updated_at': 'timestamp'}
    updated_context_flat = await service.update_context(user_id, context_data.context_data)
    
    if not updated_context_flat:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="failed to update context",
        )

    # Extract updated_at if present, otherwise default
    updated_at_str = updated_context_flat.pop("updated_at", "N/A")

    # Create the context_data dictionary for the response by removing 'updated_at'
    # The remaining items in updated_context_flat are the actual context_data
    response_context_data = updated_context_flat 

    # Construct the ContextResponse correctly
    return ContextResponse(
        user_id=user_id,
        context_data=response_context_data, # This is correct for PUT
        updated_at=updated_at_str
    )


@router.get("/{user_id}", response_model=ContextResponse)
async def get_user_context(
    user_id: UserID, redis_client: redis.Redis = Depends(get_redis_client)
):
    """Retrieve the current dynamic context for a specific user."""
    logger.info(f"Attempting to retrieve context for user_id:{user_id}")
    service = ContextInjectorService(redis_client=redis_client)
    
    # context_flat will be the flat dictionary from Redis
    context_flat = await service.get_current_context(user_id)
    
    if not context_flat:
        logger.info(f"No active context found for user_id:{user_id},returning empty.")
        # This branch correctly returns ContextResponse with context_data={}
        return ContextResponse(user_id=user_id, context_data={}, updated_at="N/A")
    
    # --- CRITICAL FIX FOR GET ENDPOINT STARTS HERE ---
    # Extract updated_at, using .pop() to remove it from context_flat
    updated_at_str = context_flat.pop("updated_at", "N/A")

    # The remaining items in context_flat are the actual dynamic context_data
    response_context_data = context_flat 

    logger.info(f"Retrieved context for user_id:{user_id}")
    # Construct ContextResponse by explicitly assigning to context_data field
    return ContextResponse(
        user_id=user_id,
        context_data=response_context_data, # This is the crucial change for GET
        updated_at=updated_at_str
    )
    # --- CRITICAL FIX FOR GET ENDPOINT ENDS HERE ---


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_context(
    user_id: UserID, redis_client: redis.Redis = Depends(get_redis_client)
):
    """Delete all dynamic context data for a specific user."""
    logger.info(f"Attempting to delete context for user_id: {user_id}")
    service = ContextInjectorService(redis_client)
    deleted = await service.delete_context(user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Context not found for deletion",
        )
    logger.info(f"Context deleted for user_id: {user_id}")
    return