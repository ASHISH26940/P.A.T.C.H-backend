# app/api/v1/endpoints/chat.py

from fastapi import APIRouter, Depends, HTTPException, status
from app.models.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.services.vector_store import VectorStoreService
from app.services.context_injector import ContextInjectorService
from loguru import logger

# Create a FastAPI APIRouter instance for chat-related endpoints
router = APIRouter()

# --- Dependencies for injecting services into endpoints ---

# Dependency for VectorStoreService
# This function will be called by FastAPI to provide an instance of VectorStoreService
# when an endpoint that depends on it is called.
async def get_vector_store_service() -> VectorStoreService:
    # In a real application, you might pass configuration or a database client here
    # For now, we'll assume VectorStoreService can be initialized directly or has its own dependencies handled internally.
    # If it needs a ChromaDB client, ensure it's passed here or managed within VectorStoreService's __init__.
    # Example (assuming chroma_client is available via another dependency or global/singleton):
    from app.core.chroma_client import get_chroma_client
    return VectorStoreService(chroma_client=get_chroma_client())


# Dependency for ContextInjectorService
# This will provide an instance of ContextInjectorService, which needs a Redis client.
async def get_context_injector_service() -> ContextInjectorService:
    from app.core.redis_client import get_redis_client
    return ContextInjectorService(redis_client=await get_redis_client())


# Dependency for ChatService
# This ensures ChatService is instantiated with its required dependencies.
async def get_chat_service(
    vector_store_service: VectorStoreService = Depends(get_vector_store_service),
    context_injector_service: ContextInjectorService = Depends(get_context_injector_service)
) -> ChatService:
    return ChatService(
        vector_store_service=vector_store_service,
        context_injector_service=context_injector_service
    )


# --- API Endpoint Definition ---

@router.post("/", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_with_ai(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
) -> ChatResponse:
    """
    Handles chat messages from the user, processes them with AI (RAG and LLM),
    and returns an AI-generated response.
    """
    logger.info(f"Received chat request from user '{request.user_id}' for collection '{request.collection_name}'")
    
    try:
        response = await chat_service.process_chat_message(request)
        logger.info(f"Successfully processed chat request for user '{request.user_id}'.")
        return response
    except ValueError as ve:
        logger.error(f"Chat processing failed due to configuration error: {ve}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration error during chat processing: {ve}"
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during chat processing for user '{request.user_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request. Please try again later."
        )

