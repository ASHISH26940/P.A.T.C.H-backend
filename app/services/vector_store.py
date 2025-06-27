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
        self.embedder = get_embeddings_model() # This should return the actual embedder instance

    async def get_or_create_collection(
        self, collection_name: str, collection_metadata: Optional[Dict[str, Any]] = None
    ) -> Collection:
        """Gets an existing collection or create a new one."""
        try:
            # ChromaDB handles embedding automatically if you initialize it with an embedder,
            # or if it has a default. If you use a custom embedder like GoogleGenerativeAIEmbeddings,
            # you might need to pass it explicitly when creating the collection or during add/query.
            # For HttpClient, typically the embedder is configured server-side or on the client init.
            # Assuming current setup of get_embeddings_model is sufficient for the client to use.
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name, metadata=collection_metadata # You can pass embedder here too if needed: embedding_function=self.embedder
            )
            logger.info(f"Accessed/Created ChromaDB collection: {collection_name}")
            return collection
        except Exception as e:
            logger.error(f"Failed to get/create collection {collection_name}: {e}")
            raise

    async def add_documents(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        """Adds a list of Document objects to a specified collection."""
        if not documents:
            return []

        collection = await self.get_or_create_collection(
            collection_name=collection_name
        )
        ids = [doc.id if doc.id else str(uuid.uuid4()) for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        try:
            # Explicitly generate embeddings if the client or collection doesn't auto-embed
            # This is safer to ensure embeddings are created before adding to Chroma.
            embeddings_list = self.embedder.embed_documents(contents) # <--- Generate embeddings explicitly

            collection.add(
                documents=contents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings_list # <--- Pass embeddings
            )
            logger.info(
                f"Added {len(documents)} documents to collection '{collection_name}'."
            )
            return ids
        except Exception as e:
            logger.error(
                f"Failed to add documents to collection '{collection_name}': {e}"
            )
            raise

    async def add_document_to_collection( # <--- NEW METHOD
        self,
        collection_name: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Adds a single document (content + metadata) to a specified collection.
        Automatically generates ID if not provided.
        """
        document = Document(
            id=doc_id if doc_id else str(uuid.uuid4()),
            content=content,
            metadata=metadata if metadata is not None else {}
        )
        # Use the existing add_documents method to handle the actual addition
        added_ids = await self.add_documents(collection_name=collection_name, documents=[document])
        return added_ids[0] if added_ids else None # Return the ID of the added document

    async def query_documents(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: List[str] = ["documents", "metadatas", "distances"],
        min_similarity_score: Optional[float] = None # <--- NEW PARAMETER for filtering
    ) -> List[DocumentQueryResult]:
        """
        Queries a collection with a given text and returns relevant documents.
        Supports filtering by minimum similarity score.
        """

        if not query_text:
            return []

        collection = await self.get_or_create_collection(
            collection_name=collection_name
        )

        try:
            # Embed the query text using the instantiated embedder
            query_embeddings = self.embedder.embed_query(query_text) # <--- Embed query explicitly

            results = collection.query(
                query_embeddings=[query_embeddings], # <--- Pass embeddings directly
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include,
            )

            query_result_list: List[DocumentQueryResult] = []

            # Ensure lists are not None before accessing elements
            current_ids = results.get("ids", [[]])[0]
            current_documents = results.get("documents", [[]])[0]
            current_metadatas = results.get("metadatas", [[]])[0]
            current_distances = results.get("distances", [[]])[0]

            for j in range(len(current_ids)):
                doc_id = current_ids[j]
                doc_content = current_documents[j]
                meta = current_metadatas[j]
                dist = current_distances[j] # This is distance, so lower is better similarity

                # If min_similarity_score is provided, filter results
                # Note: ChromaDB distances are L2 (Euclidean) by default for embedding-001,
                # where 0 is perfect match and larger numbers mean less similar.
                # If your embedder/Chroma uses cosine similarity, it's typically 1 for perfect match
                # and -1 for completely dissimilar. Adjust threshold comparison accordingly.
                # Assuming L2, so a lower distance means higher similarity.
                # If min_similarity_score is a *similarity* (higher is better), we need to map distance to similarity.
                # For L2 distance, a higher similarity means a *lower* distance.
                # Let's assume min_similarity_score is a threshold on the *distance*,
                # meaning we only keep documents where distance <= min_similarity_score.
                # If `min_similarity_score` is conceptually a *similarity* score (e.g., cosine similarity 0-1),
                # you'd need to convert distance to similarity or adjust the logic.
                # For `embedding-001`, L2 distance is common.
                # A typical way to conceptualize a "similarity threshold" with L2 distance
                # is to define a maximum acceptable distance. Let's rename for clarity.

                # For now, let's interpret `min_similarity_score` as the *maximum acceptable distance*.
                # If you need it to be a 0-1 similarity score, you'll need to define a distance-to-similarity function.
                if min_similarity_score is not None:
                    # Convert ChromaDB's distance (0=identical, larger=dissimilar)
                    # to a similarity score (1=identical, smaller=dissimilar)
                    # Assuming distance is in [0, 1] range for normalized cosine distance
                    calculated_similarity = 1 - dist

                    if calculated_similarity < min_similarity_score:
                        logger.debug(
                            f"Skipping document {id} due to calculated similarity {calculated_similarity:.4f} "
                            f"being less than threshold {min_similarity_score:.4f} (distance was {dist:.4f})"
                        )
                        continue
                
                query_result_list.append(
                    DocumentQueryResult(
                        document=Document(
                            id=doc_id,
                            content=doc_content,
                            metadata=meta if meta is not None else {}
                        ),
                        distance=dist # Keep distance, but it represents dissimilarity
                    )
                )

            logger.info(
                f"Queried collection '{collection_name}'. Found {len(query_result_list)} results after filtering."
            )
            return query_result_list
        except Exception as e:
            logger.error(f"Failed to query collection '{collection_name}': {e}", exc_info=True)
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