import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from typing import List, Dict, Any, Optional
from loguru import logger
import uuid

from app.models.document import Document, DocumentCollectionCreate, DocumentQueryResult
from app.core.llm_client import get_embeddings_model


class VectorStoreService:
    def __init__(self, chroma_client: ClientAPI):
        self.chroma_client = chroma_client
        # FIX 1: Call get_embeddings_model() to get the actual model instance
        self.embedder = get_embeddings_model() 

    async def get_or_create_collection(
        self, collection_name: str, collection_metadata: Optional[Dict[str, Any]] = None
    ) -> Collection:
        """Gets an existing collection or create a new one."""
        try:
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name, metadata=collection_metadata
            )
            logger.info(f"Accessed/Created ChromaDB collection: {collection_name}")
            return collection
        except Exception as e:
            logger.error(f"Failed to get/create collection {collection_name}: {e}")
            raise

    async def add_documents(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        """Adds documents to a specified collection."""
        if not documents:
            return []

        collection = await self.get_or_create_collection(
            collection_name=collection_name
        )
        ids = [doc.id if doc.id else str(uuid.uuid4()) for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        try:
            # ChromaDB add requires embeddings if no default embedder is set or provided
            # Assuming `self.chroma_client` or the collection has an embedder,
            # or `add` can infer from contents with a default.
            # If you run into issues here, you might need to call self.embedder.embed_documents(contents)
            # and pass `embeddings=embeddings_list` to collection.add().
            collection.add(documents=contents, metadatas=metadatas, ids=ids)
            logger.info(
                f"Added {len(documents)} documents to collection '{collection_name}'."
            )
            return ids
        except Exception as e:
            logger.error(
                f"Failed to add documents to collection '{collection_name}':'{e}'"
            )
            raise

    # FIX 2: Change query_texts to query_text (single string) and return type to List[DocumentQueryResult]
    async def query_documents(
        self,
        collection_name: str,
        query_text: str, # Changed from query_texts: List[str]
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: List[str] = ["documents", "metadatas", "distances"],
    ) -> List[DocumentQueryResult]: # Changed return type to List[DocumentQueryResult]
        """
        Queries a collection with a given text and returns relevant documents.
        Returns a single list of DocumentQueryResult.
        """

        if not query_text:
            return []

        collection = await self.get_or_create_collection(
            collection_name=collection_name
        )

        try:
            # Pass query_text as a list to ChromaDB's query method
            results = collection.query(
                query_texts=[query_text], # Wrap single query_text in a list
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include,
            )
            
            # Since we pass a single query_text, we expect only one inner list of results
            # Access the first (and only) element of the outer lists
            query_result_list: List[DocumentQueryResult] = []

            current_ids = results.get("ids", [[]])[0] # Access first element
            current_documents = results.get("documents", [[]])[0]
            current_metadatas = results.get("metadatas", [[]])[0]
            current_distances = results.get("distances", [[]])[0]

            for j in range(len(current_ids)):
                doc_id = current_ids[j]
                doc_content = current_documents[j]
                meta = current_metadatas[j]
                dist = current_distances[j]

                query_result_list.append(
                    DocumentQueryResult(
                        document=Document(
                            id=doc_id,
                            content=doc_content,
                            metadata=meta if meta is not None else {}
                        ),
                        distance=dist
                    )
                )

            logger.info(
                f"Queried collection '{collection_name}' with text. Found {len(query_result_list)} results."
            )
            return query_result_list # Return a single list now
        except Exception as e:
            logger.error(f"Failed to query collection '{collection_name}': {e}")
            raise
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Deletes a collection."""
        try:
            self.chroma_client.delete_collection(name=collection_name)
            logger.info(f"Collection '{collection_name}' deleted.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            raise

    async def delete_documents(self, collection_name: str, ids: Optional[List[str]] = None, where: Optional[Dict[str, Any]] = None) -> bool:
        """Deletes documents from a collection by IDs or metadata."""
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
            collection.delete(ids=ids, where=where)
            logger.info(f"Deleted documents from '{collection_name}'. IDs: {ids}, Where: {where}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents from '{collection_name}': {e}")
            raise