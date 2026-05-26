from fastapi import FastAPI
from contextlib import asynccontextmanager
from loguru import logger
from app.core.llm_client import get_chat_llm, get_embeddings_model
from app.api.v1.router import api_router
from app.core.database import init_db
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("P.A.T.C.H starting up...")

    try:
        get_chat_llm()
        get_embeddings_model()
        logger.info("LLM client verified.")
    except ValueError as e:
        logger.error(f"LLM client startup failed: {e}")

    try:
        await init_db()
        logger.info("Database initialized (pgvector + tables).")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    yield

    logger.info("P.A.T.C.H shutting down...")


app = FastAPI(
    title="P.A.T.C.H",
    description="Creator memory infrastructure API.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/v1")

