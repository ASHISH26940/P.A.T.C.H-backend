import json
import re
from typing import List
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.models.memory import ChatMessage
from app.core.llm_client import get_chat_llm
from app.schemas.persona import DerivedPersonaSuggestion

DERIVE_PROMPT = """You are analyzing a transcript of chat messages from a video creator's assistant session. Your job is to identify distinct "personas" — recurring roles, topics, or modes the user adopts while working.

Given the chat history below, identify up to 3 distinct personas. A persona is defined by:
- A consistent topic domain (e.g. "video editing", "script writing", "channel analytics")
- A recurring tone or role the user takes (e.g. "strategic planner", "hands-on editor", "critic")
- Distinct goals and traits that emerge from their questions and instructions

For each persona, return a JSON object with:
- name: short, memorable title (2-4 words)
- description: what this persona represents, why it emerged (1-2 sentences)
- traits: list of 2-4 characteristic traits
- goals: list of 2-3 goals this persona seems to pursue
- confidence: 0.0-1.0 score based on how clearly defined this cluster is
- sample_message_indices: list of 0-based indices into the messages array that best represent this persona (max 3)

Return ONLY a JSON array. No markdown, no explanation.

Example:
[
  {
    "name": "Technical Reviewer",
    "description": "Focuses on code quality, deployment architecture, and tooling decisions.",
    "traits": ["analytical", "detail-oriented", "skeptical"],
    "goals": ["ensure production stability", "optimize performance"],
    "confidence": 0.85,
    "sample_message_indices": [2, 7, 14]
  }
]

Chat messages (indexed 0-based, newest first):
"""


class DerivedPersonaService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.llm = get_chat_llm()

    async def derive_personas(
        self, user_id: str, min_messages: int = 10
    ) -> List[DerivedPersonaSuggestion]:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.user_id == user_id, ChatMessage.role == "human")
            .order_by(desc(ChatMessage.created_at))
            .limit(100)
        )
        result = await self.db.execute(stmt)
        messages: List[ChatMessage] = list(result.scalars().all())

        if len(messages) < min_messages:
            logger.info(
                f"Not enough messages ({len(messages)}/{min_messages}) for user {user_id} to derive personas"
            )
            return []

        message_texts = [m.content for m in messages]
        numbered = "\n".join(
            f"[{i}] {t}" for i, t in enumerate(message_texts)
        )

        try:
            response = await self.llm.ainvoke(DERIVE_PROMPT + numbered)
            content = response.content.strip()
            content = re.sub(r"^```\w*\n?", "", content)
            content = re.sub(r"\n```$", "", content)
            content = content.strip()
            clusters = json.loads(content)
            if not isinstance(clusters, list):
                logger.warning(f"LLM returned non-list JSON: {type(clusters)}")
                return []
        except Exception as e:
            logger.error(f"LLM persona derivation failed: {e}")
            return []

        suggestions = []
        for c in clusters:
            sample_indices = c.get("sample_message_indices", [])
            samples = []
            for idx in sample_indices:
                if 0 <= idx < len(message_texts):
                    t = message_texts[idx]
                    if len(t) > 200:
                        t = t[:200] + "..."
                    samples.append(t)

            suggestions.append(
                DerivedPersonaSuggestion(
                    name=c.get("name", "Unnamed"),
                    description=c.get("description", ""),
                    traits=c.get("traits", []),
                    goals=c.get("goals", []),
                    confidence=min(max(float(c.get("confidence", 0.5)), 0.0), 1.0),
                    sample_messages=samples,
                    message_count=len(sample_indices),
                )
            )

        return suggestions
