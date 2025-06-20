from fastapi import FastAPI
from contextlib import asynccontextmanager
from loguru import logger
from app.core.config import settings
from app.core.llm_client import get_chat_llm,get_embeddings_model
from app.api.v1.router import api_router
from app.core.database import init_db
from app.core.redis_client import init_redis_client, close_redis_client
from app.core.chroma_client import init_chroma_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Initializes and closes connections to external services.
    """
    logger.info("CogniFlow AI Service starting up (Lifespan event)...")
    
    # --- Startup Logic (formerly @app.on_event("startup")) ---
    # Verify LLM client initialization
    try:
        get_chat_llm()
        get_embeddings_model()
        logger.info("LLM client connectivity verified (basic check).")
    except ValueError as e:
        logger.error(f"LLM client startup failed: {e}")
    
    # Initialize database
    try:
        await init_db()
        logger.info("Database startup initialization complete.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    # Initialize Redis client
    try:
        await init_redis_client()
        logger.info("Redis client startup initialization complete.")
    except Exception as e:
        logger.error(f"Redis client initialization failed: {e}")
        # Depending on severity, you might want to exit or disable Redis-dependent features

    # Initialize ChromaDB client
    try:
        init_chroma_client() # This is synchronous, no await needed if HttpClient
        logger.info("ChromaDB client startup initialization complete.")
    except Exception as e:
        logger.error(f"ChromaDB client initialization failed: {e}")
        # Consider if this should prevent startup if ChromaDB is critical

    yield # This line separates startup from shutdown

    # --- Shutdown Logic (formerly @app.on_event("shutdown")) ---
    logger.info("CogniFlow AI Service shutting down (Lifespan event)...")
    await close_redis_client()
    # No explicit close for HttpClient usually for ChromaDB.
    # If you implemented one for a persistent client: await close_chroma_client()


app=FastAPI(
    title="P.A.T.C.H",
    description="Internal API SERVICE for memory and context management.",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router,prefix="/v1")

