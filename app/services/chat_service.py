import uuid
import re
from typing import Optional
from loguru import logger
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from app.models.chat import ChatRequest, ChatResponse
from app.core.llm_client import get_chat_llm
from app.services.memory_service import MemoryService
from app.core.config import settings


class ChatService:
    def __init__(
        self,
        memory_service: MemoryService,
        llm: Optional[BaseChatModel] = None,
    ):
        self.memory_service = memory_service
        self.llm = llm if llm else get_chat_llm()
        self.output_parser = StrOutputParser()

        self.system_prompt_template = """
You are a helpful AI assistant for video creators. Be concise and specific.

{memory_context}

After your response, if the user shared something worth remembering later, add each fact on its own line starting with 📝. Example:
Your normal response here.
📝 User prefers voiceover-style tutorials
📝 They're working on a gaming channel
"""

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt_template),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        self.chain = self.prompt | self.llm | self.output_parser

    async def process_chat_message(self, request: ChatRequest) -> ChatResponse:
        user_id = request.user_id
        user_message = request.user_message
        message_id = str(uuid.uuid4())

        combined = []
        memory_context = ""
        try:
            important_memories = await self.memory_service.get_important_memories(
                user_id=user_id, limit=3
            )
            relevant_memories = await self.memory_service.search_memories(
                user_id=user_id,
                query=user_message,
                n_results=settings.MEMORY_N_RESULTS,
                min_similarity=settings.MEMORY_SIMILARITY_THRESHOLD,
            )

            seen_ids = {m.id for m in relevant_memories}
            combined = list(relevant_memories)
            for m in important_memories:
                if m.id not in seen_ids:
                    combined.append(m)
                    seen_ids.add(m.id)

            memory_context_parts = []
            for mem in combined:
                memory_context_parts.append(f"[{mem.memory_type}] {mem.content}")
            memory_context = (
                "\n".join(memory_context_parts) if memory_context_parts else ""
            )
        except Exception as e:
            logger.warning(f"Memory retrieval failed (continuing without context): {e}")

        chat_history = await self.memory_service.get_chat_history(
            user_id=user_id, limit=50
        )

        system_message = SystemMessage(
            content=self.system_prompt_template.format(
                memory_context=memory_context
            )
        )

        messages_for_llm = [system_message]

        token_budget = int(
            settings.LLM_CONTEXT_WINDOW
            * settings.LLM_TOKEN_THRESHOLD_PERCENTAGE
            - settings.LLM_RESERVED_TOKENS_OUTPUT
        )
        system_tokens = len(system_message.content)
        available = token_budget - system_tokens

        trimmed_history = []
        for msg in reversed(chat_history):
            msg_tokens = len(msg.content)
            if available - msg_tokens > 0:
                if msg.role == "human":
                    trimmed_history.insert(0, HumanMessage(content=msg.content))
                elif msg.role == "ai":
                    trimmed_history.insert(0, AIMessage(content=msg.content))
                available -= msg_tokens
            else:
                break

        messages_for_llm.extend(trimmed_history)
        messages_for_llm.append(HumanMessage(content=user_message))

        try:
            ai_response = await self.chain.ainvoke({
                "messages": messages_for_llm,
                "memory_context": memory_context,
            })
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            ai_response = "Sorry, I encountered an error processing your request."

        display_response = ai_response
        extractions = re.findall(r"^📝 (.+)$", ai_response, re.MULTILINE)
        if extractions:
            display_response = re.sub(r"\n?📝 .+", "", ai_response).strip()

            for item in extractions:
                try:
                    await self.memory_service.add_memory(
                        user_id=user_id, content=item.strip(),
                        memory_type="extraction", importance=0.6,
                        metadata={"source_message_id": message_id},
                    )
                except Exception as e:
                    logger.debug(f"Failed to store extraction: {e}")

            try:
                await self.memory_service.add_extraction(
                    user_id=user_id, session_id=message_id,
                    source="chat", insights={"extractions": extractions},
                )
                logger.info(f"Stored {len(extractions)} extractions from {message_id}")
            except Exception as e:
                logger.debug(f"Failed to log extraction event: {e}")

        await self.memory_service.add_chat_message(
            user_id=user_id, role="human", content=user_message
        )
        await self.memory_service.add_chat_message(
            user_id=user_id, role="ai", content=display_response
        )

        await self.memory_service.add_memory(
            user_id=user_id,
            content=f"Q: {user_message}\nA: {display_response}",
            memory_type="qa",
            metadata={"message_id": message_id},
            importance=0.3,
        )

        return ChatResponse(
            ai_response=display_response,
            source_documents=[
                {"id": m.id, "content": m.content, "metadata": m.metadata_,
                 "type": m.memory_type, "importance": m.importance}
                for m in combined
            ],
            message_id=message_id,
        )
