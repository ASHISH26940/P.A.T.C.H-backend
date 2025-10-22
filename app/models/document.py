# app/models/document.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Document(BaseModel):
    id: Optional[str] = Field(None, description="Unique identifier for the document. If not provided, ChromaDB will generate one.")
    content: str = Field(..., min_length=1, description="The main text content of the document.")
    metadata: Dict[str, Any] = Field({}, description="Arbitrary key-value pairs for filtering and organization.")

class DocumentCollectionCreate(BaseModel):
    name: str = Field(..., min_length=1, description="Unique name for the document collection.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for the collection.")

class DocumentQueryResult(BaseModel):
    document: Document
    distance: Optional[float] = Field(None, description="Similarity distance (lower is more similar).")

class DocumentsAddedResponse(BaseModel):
    collection_name: str
    added_count: int
    ids: List[str]

class DocumentsQueryResponse(BaseModel):
    collection_name: str
    query_text: str
    results: List[DocumentQueryResult]

