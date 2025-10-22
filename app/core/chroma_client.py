import chromadb
from chromadb.api import ClientAPI
from app.core.config import settings
from loguru import logger

_chroma_client:ClientAPI=None

def get_chroma_client() -> ClientAPI:
    """Dependency to get a ChromaDB client instance."""
    if _chroma_client is None:
        logger.error("ChromaDB client not initialized.")
        raise ConnectionError("ChromaDB client not initialized.")
    return _chroma_client

def init_chroma_client():
    """Initialize the chromadb Client"""
    global _chroma_client
    try:
        _chroma_client=chromadb.HttpClient(
            host=settings.CHROMADB_HOST,
            port=settings.CHROMADB_PORT
        )
        logger.info(f"ChromaDb successfully initialized")
    except Exception as e:
        logger.error(f"Failed to connect to the chroma db:{e}")
        _chroma_client=None