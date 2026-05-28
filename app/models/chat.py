from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# Re-using (or defining if not already present) for source_documents
class DocumentQueryResult(BaseModel):
    id: str
    content: str
    metadata: Dict
    distance: Optional[float] = None # Optional, if your Chroma query returns it

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user or model)")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    user_message: str = Field(..., description="The current message from the user")
    collection_name: str = Field(..., description="The ChromaDB collection to query for context")
    user_id: str = Field(..., description="Unique identifier for the user/conversation session")
    session_id: Optional[str] = Field(
        None, description="Chat session UUID for grouping messages by conversation."
    )
    past_messages: Optional[List[ChatMessage]] = Field(
        None, description="Recent conversation history provided by the frontend via IndexedDB."
    )
    persona_id: Optional[str] = Field(
        None, description="UUID of the active persona for this chat session."
    )
    persona_name: Optional[str] = Field(
        None, description="Name of the active persona for system prompt customization."
    )

class ChatResponse(BaseModel):
    ai_response: str = Field(..., description="The AI's generated response")
    source_documents: Optional[List[DocumentQueryResult]] = Field(
        None, description="Relevant documents retrieved from ChromaDB that informed the answer."
    )
    message_id: str = Field(..., description="Unique identifier for this AI response.")
    derivation_available: bool = Field(
        False, description="True when enough chat history exists to derive personas."
    )