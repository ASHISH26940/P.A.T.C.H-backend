# app/services/chat_service.py

from typing import Dict, Any, List, Optional
from loguru import logger
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from app.models.chat import ChatRequest, ChatResponse
from app.models.document import Document, DocumentQueryResult # Ensure this is imported
from app.core.llm_client import get_chat_llm
from app.services.vector_store import VectorStoreService
from app.services.context_injector import ContextInjectorService
from app.core.config import settings # Import settings for thresholds
import uuid
from datetime import datetime,timezone

class ChatService:
    def __init__(
        self,
        vector_store_service: VectorStoreService,
        context_injector_service: ContextInjectorService,
        llm: Optional[BaseChatModel] = None,
    ):
        self.vector_store_service = vector_store_service
        self.context_injector_service = context_injector_service
        self.llm = llm if llm else get_chat_llm()
        self.output_parser = StrOutputParser()

        # Define the system prompt for the LLM
        self.system_prompt_template = """
        You are a helpful and knowledgeable AI assistant. Your goal is to provide accurate and concise answers based on the provided context.
        If the answer cannot be found in the context, politely state that you don't know or that the information is not available in the provided documents.
        Avoid making up information. Maintain a consistent persona. Be concise and to the point.

        {rag_context}
        {history_context}
        {cognitive_context}
        """

        # Initialize the prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt_template),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # Build the RAG chain (simplified for this example)
        self.rag_chain = self.prompt | self.llm | self.output_parser

    async def process_chat_message(self, request: ChatRequest) -> ChatResponse:
        user_id = request.user_id
        user_message_content = request.user_message
        collection_name = request.collection_name
        message_id = str(uuid.uuid4()) # Generate a unique ID for this message

        logger.info(f"Processing chat message for user '{user_id}' in collection '{collection_name}'")

        all_source_documents_for_response: List[DocumentQueryResult] = []
        rag_context_parts = []
        
        # --- Layer 1: User's Past Questions & Answers (Vector Store) ---
        user_qa_collection_name = f"user_past_questions_answers"
        try:
            logger.debug(f"Attempting to retrieve past Q&A for user '{user_id}' with threshold {settings.CHROMA_PAST_QA_SIMILARITY_THRESHOLD}")
            past_qa_documents = await self.vector_store_service.query_documents(
                user_qa_collection_name,
                user_message_content,
                n_results=settings.CHROMA_PAST_QA_N_RESULTS,
                min_similarity_score=settings.CHROMA_PAST_QA_SIMILARITY_THRESHOLD
            )
            if past_qa_documents:
                logger.info(f"Found {len(past_qa_documents)} past Q&A documents for user '{user_id}'.")
                rag_context_parts.append("### Relevant Past Interactions:\n")
                for doc in past_qa_documents:
                    rag_context_parts.append(f"- Q: {doc.document.metadata.get('question', 'N/A')}\n  A: {doc.document.content}\n")
                all_source_documents_for_response.extend(past_qa_documents)
            else:
                logger.info("No similar past question found above threshold.")
        except Exception as e:
            logger.warning(f"Failed to retrieve past Q&A for user '{user_id}': {e}")

        # --- Layer 2: General Knowledge Base (Vector Store) ---
        try:
            logger.debug(f"Attempting to retrieve general knowledge documents for collection '{collection_name}' with threshold {settings.CHROMA_GENERAL_SIMILARITY_THRESHOLD}")
            general_documents = await self.vector_store_service.query_documents(
                collection_name,
                user_message_content,
                n_results=settings.CHROMA_GENERAL_N_RESULTS,
                min_similarity_score=settings.CHROMA_GENERAL_SIMILARITY_THRESHOLD
            )
            if general_documents:
                logger.info(f"Found {len(general_documents)} general documents for collection '{collection_name}'.")
                rag_context_parts.append("### General Knowledge:\n")
                for doc in general_documents:
                    rag_context_parts.append(f"- {doc.document.content}\n")
                all_source_documents_for_response.extend(general_documents)
            else:
                logger.info("No general documents retrieved for RAG.")
        except Exception as e:
            logger.warning(f"Failed to retrieve general knowledge for collection '{collection_name}': {e}")


        # --- Layer 3: Historical Context Retrieval (Redis/Long-term memory keywords) ---
        history_context_str = ""
        keywords_for_historical_context = ["yesterday", "last time", "previous conversation", "memory", "remember", "before"]
        if any(keyword in user_message_content.lower() for keyword in keywords_for_historical_context):
            try:
                logger.debug(f"Keywords detected for Historical Context. Attempting to retrieve long-term memory for user '{user_id}'.")
                # Assuming you have a collection for user's long-term memory/summaries
                user_long_term_memory_collection = f"user_long_term_memory"
                historical_documents = await self.vector_store_service.query_documents(
                    user_long_term_memory_collection,
                    user_message_content,
                    n_results=settings.CHROMA_HISTORY_N_RESULTS,
                    min_similarity_score=settings.CHROMA_HISTORY_SIMILARITY_THRESHOLD
                )
                if historical_documents:
                    logger.info(f"Found {len(historical_documents)} historical context documents for user '{user_id}'.")
                    history_context_str = "### Historical Context (Long-Term Memory):\n"
                    for doc in historical_documents:
                        history_context_str += f"- {doc.document.content}\n"
                    all_source_documents_for_response.extend(historical_documents)
                else:
                    logger.info("No historical context documents retrieved from long-term memory.")
            except Exception as e:
                logger.warning(f"Failed to retrieve historical context for user '{user_id}': {e}")
        else:
            logger.info("No keywords detected for Historical Context. Skipping long-term memory retrieval.")

        # --- Layer 4: Cognitive Knowledge Base (Vector Store) ---
        cognitive_context_str = ""
        try:
            logger.debug(f"Attempting to retrieve cognitive knowledge documents for user '{user_id}'.")
            cognitive_knowledge_collection = f"cognitive_knowledge_base" # A general cognitive base for all users
            cognitive_documents = await self.vector_store_service.query_documents(
                cognitive_knowledge_collection,
                user_message_content,
                n_results=settings.CHROMA_COGNITIVE_N_RESULTS,
                min_similarity_score=settings.CHROMA_COGNITIVE_SIMILARITY_THRESHOLD
            )
            if cognitive_documents:
                logger.info(f"Found {len(cognitive_documents)} cognitive knowledge documents.")
                cognitive_context_str = "### Cognitive Knowledge:\n"
                for doc in cognitive_documents:
                    cognitive_context_str += f"- {doc.document.content}\n"
                all_source_documents_for_response.extend(cognitive_documents)
            else:
                logger.info("No cognitive knowledge documents retrieved.")
        except Exception as e:
            logger.warning(f"Failed to retrieve cognitive knowledge: {e}")

        rag_context = "\n".join(rag_context_parts) if rag_context_parts else "No specific RAG context provided."
        
        # Determine if any RAG or historical/cognitive context was generated
        if not rag_context_parts and not history_context_str and not cognitive_context_str:
            logger.info("No significant RAG or historical/cognitive context generated.")


        # --- Prepare messages for LLM (including chat history from Redis) ---
        full_chat_history_from_redis = await self.context_injector_service.get_chat_history(user_id)
        logger.info(f"Fetched {len(full_chat_history_from_redis)} messages from Redis for token management.")

        # Construct messages for the LLM, ensuring token limits are respected
        messages_for_llm = []
        # The system message is constructed with the RAG/history/cognitive context
        system_message_content = self.system_prompt_template.format(
            rag_context=rag_context,
            history_context=history_context_str,
            cognitive_context=cognitive_context_str
        )
        messages_for_llm.append(SystemMessage(content=system_message_content))

        # Add historical messages, trimming if necessary
        # We need to estimate token count. A rough estimate is 4 chars per token.
        # This is a basic approach and can be refined with actual tokenizers if needed.
        system_tokens = len(system_message_content) / 4 # Rough estimate
        logger.debug(f"LLM Context Window: {settings.LLM_CONTEXT_WINDOW}, Threshold: {settings.LLM_TOKEN_THRESHOLD_PERCENTAGE}%")
        logger.debug(f"System/RAG Context Tokens: {system_tokens}")
        
        max_history_tokens = settings.LLM_CONTEXT_WINDOW * (settings.LLM_TOKEN_THRESHOLD_PERCENTAGE / 100) - system_tokens
        logger.debug(f"Max Tokens for History: {max_history_tokens}")
        
        current_history_tokens = 0
        trimmed_history = []
        for msg_dict in reversed(full_chat_history_from_redis):
            msg_content = msg_dict.get("content", "")
            msg_role = msg_dict.get("role", "")
            
            # Rough token count for this message
            msg_tokens = len(msg_content) / 4
            
            if current_history_tokens + msg_tokens <= max_history_tokens:
                if msg_role == "human":
                    trimmed_history.insert(0, HumanMessage(content=msg_content))
                elif msg_role == "ai":
                    trimmed_history.insert(0, AIMessage(content=msg_content))
                current_history_tokens += msg_tokens
            else:
                logger.warning(f"Trimming chat history to fit context window. Skipped message: {msg_content[:50]}...")
                break # Stop adding if next message exceeds limit

        messages_for_llm.extend(trimmed_history)
        # Add the current user message at the end
        messages_for_llm.append(HumanMessage(content=user_message_content))

        # Re-calculate total tokens for debug logging
        final_token_count = sum(len(m.content) for m in messages_for_llm if hasattr(m, 'content')) / 4 + (len(messages_for_llm) * 4) # Add a small buffer for message objects
        logger.debug(f"Final messages for LLM: {messages_for_llm}")
        logger.debug(f"Final token count for LLM: {final_token_count}")

        # --- Invoke LLM ---
        try:
            logger.info("Invoking LLM for response generation...")
            llm_response_message = await self.llm.ainvoke(messages_for_llm)
            ai_response_content = llm_response_message.content
            logger.info("Gemini response generated successfully using ainvoke().")
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            ai_response_content = "I am sorry, but I encountered an error while processing your request."

        # --- Store conversation turn in Redis ---
        await self.context_injector_service.add_message_to_history(
            user_id=user_id,
            role="human",
            content=user_message_content
        )
        await self.context_injector_service.add_message_to_history(
            user_id=user_id,
            role="ai",
            content=ai_response_content
        )
        logger.info(f"Conversation turn stored in Redis for user '{user_id}'.")

        # --- Store new Q&A in Past Questions & Answers collection ---
        try:
            new_qa_document = Document(
                id=str(uuid.uuid4()),
                content=ai_response_content,
                metadata={"user_id": user_id, "question": user_message_content, "timestamp": datetime.now(timezone.utc).isoformat()}
            )
            await self.vector_store_service.add_documents(
                user_qa_collection_name,
                [new_qa_document]
            )
            logger.info(f"Stored new Q&A pair in '{user_qa_collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to store Q&A pair in vector store: {e}")

        # --- Construct and return ChatResponse ---
        return ChatResponse(
            ai_response=ai_response_content,
            # This line was corrected to extract the 'document' and then dump it.
            source_documents=[doc_qr.document.model_dump() for doc_qr in all_source_documents_for_response if doc_qr.document],
            message_id=message_id
        )