# app/api/v1/endpoints/document.py
from fastapi import APIRouter, Depends, HTTPException, status
from chromadb.api import ClientAPI
from typing import List, Optional, Dict, Any
from loguru import logger

from app.core.chroma_client import get_chroma_client
from app.services.vector_store import VectorStoreService
from app.models.document import Document, DocumentCollectionCreate, DocumentsAddedResponse, DocumentsQueryResponse, DocumentQueryResult

router = APIRouter()

@router.post("/collections", response_model=DocumentCollectionCreate, status_code=status.HTTP_201_CREATED)
async def create_collection(
    collection_data: DocumentCollectionCreate,
    chroma_client: ClientAPI = Depends(get_chroma_client)
):
    """Create a new ChromaDB collection."""
    logger.info(f"Attempting to create collection: {collection_data.name}")
    service = VectorStoreService(chroma_client)
    try:
        collection = await service.get_or_create_collection(
            collection_data.name,
            collection_data.metadata
        )
        logger.info(f"Collection '{collection_data.name}' created/accessed successfully.")
        return DocumentCollectionCreate(name=collection.name, metadata=collection.metadata)
    except Exception as e:
        logger.error(f"Error creating collection {collection_data.name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create collection: {e}")

@router.post("/{collection_name}/documents", response_model=DocumentsAddedResponse, status_code=status.HTTP_201_CREATED)
async def add_documents_to_collection(
    collection_name: str,
    documents: List[Document],
    chroma_client: ClientAPI = Depends(get_chroma_client)
):
    """Add documents to a specific ChromaDB collection."""
    logger.info(f"Attempting to add {len(documents)} documents to collection '{collection_name}'.")
    service = VectorStoreService(chroma_client)
    try:
        added_ids = await service.add_documents(collection_name, documents)
        logger.info(f"Added {len(added_ids)} documents to collection '{collection_name}'.")
        return DocumentsAddedResponse(collection_name=collection_name, added_count=len(added_ids), ids=added_ids)
    except Exception as e:
        logger.error(f"Error adding documents to collection {collection_name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to add documents: {e}")

@router.post("/{collection_name}/query", response_model=List[List[DocumentQueryResult]])
async def query_documents_in_collection(
    collection_name: str,
    query_texts: List[str],
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None,
    where_document: Optional[Dict[str, Any]] = None,
    chroma_client: ClientAPI = Depends(get_chroma_client)
):
    """Query documents in a ChromaDB collection based on similarity."""
    logger.info(f"Querying collection '{collection_name}' with {len(query_texts)} texts.")
    service = VectorStoreService(chroma_client)
    try:
        results = await service.query_documents(
            collection_name=collection_name,
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        logger.info(f"Query completed for collection '{collection_name}'.")
        return results
    except Exception as e:
        logger.error(f"Error querying collection {collection_name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to query documents: {e}")

@router.delete("/collections/{collection_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    collection_name: str,
    chroma_client: ClientAPI = Depends(get_chroma_client)
):
    """Delete a ChromaDB collection."""
    logger.info(f"Attempting to delete collection: {collection_name}")
    service = VectorStoreService(chroma_client)
    try:
        await service.delete_collection(collection_name)
        logger.info(f"Collection '{collection_name}' deleted successfully.")
        return
    except Exception as e:
        logger.error(f"Error deleting collection {collection_name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete collection: {e}")

@router.delete("/{collection_name}/documents", status_code=status.HTTP_204_NO_CONTENT)
async def delete_documents_from_collection(
    collection_name: str,
    ids: Optional[List[str]] = None,
    where: Optional[Dict[str, Any]] = None,
    chroma_client: ClientAPI = Depends(get_chroma_client)
):
    """Delete documents from a ChromaDB collection by IDs or metadata."""
    if not ids and not where:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'ids' or 'where' must be provided to delete documents."
        )
    logger.info(f"Attempting to delete documents from '{collection_name}'. IDs: {ids}, Where: {where}")
    service = VectorStoreService(chroma_client)
    try:
        await service.delete_documents(collection_name, ids=ids, where=where)
        logger.info(f"Documents deleted from collection '{collection_name}'.")
        return
    except Exception as e:
        logger.error(f"Error deleting documents from {collection_name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete documents: {e}")