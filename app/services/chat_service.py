import uuid
import re
import json
from typing import AsyncGenerator
from loguru import logger
from google import genai

from app.models.chat import ChatRequest
from app.services.memory_service import MemoryService
from app.core.config import settings


client = genai.Client(api_key=settings.GEMINI_API_KEY)


def _sse(event: str, data: dict) -> str:
    return f"data: {json.dumps({'type': event, **data})}\n\n"


def _build_prompt(memory_context: str, persona_name: str = "", persona_description: str = "") -> str:
    persona_block = ""
    if persona_name:
        persona_block = f"\nYou are acting as {persona_name}."
        if persona_description:
            persona_block += f" {persona_description}"
    return f"""You are a helpful AI assistant for video creators. Be concise and specific.{persona_block}

{memory_context}

After your response, if the user shared something worth remembering later, add each fact on its own line starting with 📝. Example:
Your normal response here.
📝 User prefers voiceover-style tutorials
📝 They're working on a gaming channel"""


class ChatService:
    def __init__(self, memory_service: MemoryService):
        self.memory_service = memory_service

    async def process_chat_message_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        user_id = request.user_id
        user_message = request.user_message
        session_id = request.session_id
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

        persona_name = request.persona_name or ""
        persona_description = ""
        if request.persona_id:
            try:
                from app.models.persona import Persona
                from app.core.database import get_db
                from sqlalchemy import select
                result = await self.memory_service.db.execute(
                    select(Persona).where(Persona.id == request.persona_id)
                )
                persona = result.scalar_one_or_none()
                if persona:
                    persona_description = persona.description or ""
            except Exception:
                pass

        chat_history = await self.memory_service.get_chat_history(
            user_id=user_id, limit=50
        )

        prompt = _build_prompt(
            memory_context=memory_context,
            persona_name=persona_name,
            persona_description=persona_description,
        )

        history_parts = []
        for msg in chat_history[-20:]:
            role = "user" if msg.role == "human" else "model"
            history_parts.append(f"role: {role}\n{msg.content}")

        full_prompt = f"{prompt}\n\n{'---'.join(history_parts)}\n\nrole: user\n{user_message}"

        chunks: list[str] = []
        try:
            stream = await client.aio.models.generate_content_stream(
                model="gemma-4-26b-a4b-it",
                contents=full_prompt,
            )
            async for chunk in stream:
                if not chunk.candidates:
                    continue
                parts = chunk.candidates[0].content.parts
                for p in parts:
                    if hasattr(p, 'text') and p.text and not getattr(p, 'thought', False):
                        chunks.append(p.text)
                        yield _sse("token", {"text": p.text})
        except Exception as e:
            logger.error(f"LLM stream failed: {e}")
            yield _sse("error", {"text": "Sorry, I encountered an error processing your request."})
            return

        full_response = "".join(chunks)
        display_response = full_response

        extractions = re.findall(r"^📝 (.+)$", full_response, re.MULTILINE)
        if extractions:
            display_response = re.sub(r"\n?📝 .+", "", full_response).strip()
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
            except Exception as e:
                logger.debug(f"Failed to log extraction event: {e}")

        await self.memory_service.add_chat_message(
            user_id=user_id, role="human", content=user_message, session_id=session_id,
        )
        await self.memory_service.add_chat_message(
            user_id=user_id, role="ai", content=display_response, session_id=session_id,
        )

        await self.memory_service.add_memory(
            user_id=user_id,
            content=f"Q: {user_message}\nA: {display_response}",
            memory_type="qa",
            metadata={"message_id": message_id},
            importance=0.3,
        )

        human_count = len(chat_history) // 2 + 1

        yield _sse("done", {
            "message_id": message_id,
            "source_documents": [
                {"id": m.id, "content": m.content, "metadata": m.metadata_,
                 "type": m.memory_type, "importance": m.importance}
                for m in combined
            ],
            "derivation_available": human_count >= 10,
        })
