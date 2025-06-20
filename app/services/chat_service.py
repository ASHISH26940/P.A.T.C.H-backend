# app/services/chat_service.py

from app.core.llm_client import get_chat_llm
from app.services.vector_store import VectorStoreService
from app.services.context_injector import ContextInjectorService
from app.models.chat import ChatRequest, ChatResponse
from app.models.document import DocumentQueryResult
from typing import List, Dict, Optional, Any
import uuid
from loguru import logger

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
# from langchain_google_genai import ChatGoogleGenerativeAI # Not directly imported, but self.llm_model is this type

class ChatService:
    # Define the LLM's context window size (approximate for Gemini 1.5 Flash)
    # A safe working window, slightly less than the actual max to account for output tokens.
    LLM_CONTEXT_WINDOW = 120000 # Gemini 1.5 Flash has 128K context window
    # Threshold (percentage) at which to start truncating history
    LLM_TOKEN_THRESHOLD_PERCENTAGE = 0.90 # e.g., 90% of LLM_CONTEXT_WINDOW

    def __init__(
        self,
        vector_store_service: VectorStoreService,
        context_injector_service: ContextInjectorService
    ):
        self.vector_store_service = vector_store_service
        self.context_injector_service = context_injector_service
        self.llm_model = get_chat_llm()

    # Helper to convert internal dict format to Langchain's BaseMessage format
    def _to_langchain_message_format(self, messages: List[Dict]) -> List[BaseMessage]:
        langchain_formatted_messages = []
        for msg in messages:
            # Note: We expect messages from Redis get_chat_history to be Dicts.
            # If past_messages from ChatRequest are ever converted to a Pydantic model
            # before reaching here, adjust this logic to access msg.role, msg.content
            # This current implementation is robust for both Dict and Pydantic.
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
            else: # Fallback for objects (like Pydantic models)
                role = getattr(msg, "role", None)
                content = getattr(msg, "content", None)

            if role and content:
                if role == "user":
                    langchain_formatted_messages.append(HumanMessage(content=content))
                elif role == "model" or role == "assistant":
                    langchain_formatted_messages.append(AIMessage(content=content))
        return langchain_formatted_messages

    async def process_chat_message(self, request: ChatRequest) -> ChatResponse:
        logger.info(f"Processing chat message for user '{request.user_id}' in collection '{request.collection_name}'")

        message_id = str(uuid.uuid4())
        source_documents_for_response = []

        # --- Step 1: Retrieval Augmented Generation (RAG) ---
        rag_query_text = request.user_message
        retrieved_documents: List[DocumentQueryResult] = await self.vector_store_service.query_documents(
            query_text=rag_query_text,
            collection_name=request.collection_name, 
            n_results=4
        )
        
        document_context = ""
        if retrieved_documents:
            source_documents_for_response = retrieved_documents 
            
            doc_texts = [doc.document.content for doc in retrieved_documents if doc.document and doc.document.content]
            if doc_texts:
                # Add a clear heading for RAG context in the prompt
                document_context = "\n\n--- Start Relevant Documents ---\n" + "\n---\n".join(doc_texts) + "\n--- End Relevant Documents ---\n\n"
                logger.info(f"Retrieved {len(doc_texts)} documents for RAG.")
            else:
                logger.info("No relevant document content found for RAG.")
        else:
            logger.info("No documents retrieved for RAG.")


        # --- Step 2: Fetch Full Conversation History from Redis ---
        # Fetch ALL available chat history from Redis for comprehensive token management
        # Request.past_messages are only for initial client context, Redis is the canonical source
        full_conversation_history_from_redis: List[Dict] = await self.context_injector_service.get_chat_history(
            user_id=request.user_id,
            limit=self.context_injector_service.MAX_CHAT_HISTORY_LENGTH # Fetch up to max stored
        )
        logger.info(f"Fetched {len(full_conversation_history_from_redis)} messages from Redis for token management.")

        # Convert Redis history to Langchain's message format
        history_for_llm_processing: List[BaseMessage] = self._to_langchain_message_format(full_conversation_history_from_redis)

        # Append the *current* user message to the history for token calculation
        # This will be the last message in the list
        current_user_message_lc = HumanMessage(content=request.user_message)
        history_for_llm_processing.append(current_user_message_lc)


        # --- Step 3: Construct System Instruction and Truncate History for LLM Prompt ---
        # System instruction and RAG context should ideally be combined and static,
        # or prepended to the first message, or handled as a SystemMessage.
        # For invoke, we usually prepend it to the current user query or have a dedicated
        # system message at the start if the LLM supports it without affecting turn-taking.

        # Let's count tokens for system instruction and RAG context first.
        # This part of the prompt is relatively static or derived from RAG.
        # We'll prepend it to the *current* user message's content before sending to LLM.
        system_instruction = (
            "You are a helpful and knowledgeable AI assistant. "
            "Your goal is to provide accurate and concise answers based on the provided context. "
            "If the answer cannot be found in the context, politely state that you don't know "
            "or that the information is not available in the provided documents. "
            "Avoid making up information."
        )
        
        # Combine system instruction and RAG context
        initial_prompt_context = f"{system_instruction}{document_context}"
        
        # Calculate maximum allowed tokens for chat history, considering context window and RAG/system parts.
        # We need to count tokens of the `initial_prompt_context` and then subtract it from the LLM_CONTEXT_WINDOW.
        # Note: Langchain's `get_num_tokens_from_messages` doesn't directly take a string.
        # For system/RAG context that isn't part of the turn-by-turn history,
        # we can put it as the content of the initial user message.
        # Let's count tokens with a placeholder `HumanMessage` to estimate.
        
        # Temporarily create a list including only the system/RAG context for token counting
        temp_system_rag_message_list = [HumanMessage(content=initial_prompt_context)]
        system_rag_tokens = self.llm_model.get_num_tokens_from_messages(temp_system_rag_message_list)
        
        # Ensure we leave enough space for the current user's message itself and potential AI response tokens
        # A conservative estimate for the current message and response could be 1000-2000 tokens,
        # depending on expected verbosity. Let's reserve some buffer.
        RESERVED_TOKENS_FOR_CURRENT_TURN_AND_OUTPUT = 2000 

        max_tokens_for_history = (self.LLM_CONTEXT_WINDOW * self.LLM_TOKEN_THRESHOLD_PERCENTAGE) \
                               - system_rag_tokens \
                               - RESERVED_TOKENS_FOR_CURRENT_TURN_AND_OUTPUT

        logger.debug(f"LLM Context Window: {self.LLM_CONTEXT_WINDOW}, Threshold: {self.LLM_TOKEN_THRESHOLD_PERCENTAGE*100}%")
        logger.debug(f"System/RAG Context Tokens: {system_rag_tokens}")
        logger.debug(f"Max Tokens for History: {max_tokens_for_history}")


        # Truncate history if it exceeds the calculated max_tokens_for_history
        current_history_tokens = self.llm_model.get_num_tokens_from_messages(history_for_llm_processing)
        logger.debug(f"Current full history (incl. current user message) tokens: {current_history_tokens}")
        
        # If the current total (including the new message) already exceeds the overall limit after RAG/System
        # we need to truncate from the beginning of `history_for_llm_processing`
        
        # This loop removes oldest messages until token count is within limits
        # It's crucial to remove from the *start* (oldest) of the history list
        # since the most recent messages are usually more relevant.
        while current_history_tokens > max_tokens_for_history and len(history_for_llm_processing) > 2: # Keep at least user/model for basic convo
            # Remove the oldest message (first element)
            history_for_llm_processing.pop(0) 
            current_history_tokens = self.llm_model.get_num_tokens_from_messages(history_for_llm_processing)
            logger.debug(f"History truncated. New token count: {current_history_tokens}")

        if current_history_tokens > max_tokens_for_history:
             logger.warning(f"Despite truncation, history still exceeds max_tokens_for_history. "
                            f"Remaining tokens: {current_history_tokens} > {max_tokens_for_history}. "
                            f"This might indicate too little space for history or a very long current query/context.")


        # Final prompt construction:
        # We inject the system_and_rag_context into the *first* HumanMessage in the history.
        # If history_for_llm_processing is empty (very first message), it means current_user_message_lc is the first.
        # If it's not empty, it contains the truncated history. The last message is always the current user's.
        
        final_messages_for_llm = []
        if history_for_llm_processing:
            # Create a new list for the final messages to send to the LLM
            # The first message (oldest in the truncated history) gets the combined context
            first_message_content = history_for_llm_processing[0].content
            final_messages_for_llm.append(HumanMessage(content=f"{initial_prompt_context}\n\n{first_message_content}"))
            
            # Append the rest of the truncated history
            final_messages_for_llm.extend(history_for_llm_processing[1:])
        else:
            # If no history, just send the current user message with context
            final_messages_for_llm.append(HumanMessage(content=f"{system_and_rag_context}\n\nUser Query: {request.user_message}"))

        logger.debug(f"Final messages for LLM: {final_messages_for_llm}")
        logger.debug(f"Final token count for LLM: {self.llm_model.get_num_tokens_from_messages(final_messages_for_llm)}")


        # --- Step 4: Call Gemini using ainvoke() ---
        ai_response_content = "I'm sorry, I couldn't generate a response."
        try:
            gemini_response_message = await self.llm_model.ainvoke(
                final_messages_for_llm # Use the token-managed messages
            )
            ai_response_content = gemini_response_message.content
            logger.info("Gemini response generated successfully using ainvoke().")
        except Exception as e:
            logger.error(f"Error generating content from Gemini: {e}", exc_info=True)
            ai_response_content = "I apologize, but I encountered an error trying to generate a response. Please try again later."


        # --- Step 5: Store the Full Current Turn into Redis ---
        await self.context_injector_service.add_message_to_history(
            user_id=request.user_id,
            role="user",
            content=request.user_message
        )
        await self.context_injector_service.add_message_to_history(
            user_id=request.user_id,
            role="model",
            content=ai_response_content
        )
        logger.info(f"Conversation turn stored in Redis for user '{request.user_id}'.")


        return ChatResponse(
            ai_response=ai_response_content,
            source_documents=source_documents_for_response,
            message_id=message_id
        )