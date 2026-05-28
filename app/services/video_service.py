import subprocess
import json
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.llm_client import get_chat_llm
from app.services.memory_service import MemoryService
from langchain_core.messages import HumanMessage, SystemMessage


class VideoIngestionError(Exception):
    pass


SYSTEM_PROMPT = """You are a video analysis assistant for creators. Given a video's title, channel, duration, and description, extract EVERY actionable insight the creator would want to remember. Be exhaustive — extract as many useful pieces as you can find.

For each insight, output a line starting with 📝 followed by:
- The insight content (what's worth remembering, be specific)
- The type of insight (tip, idea, reference, technique, tool, quote, concept, framework, statistic, principle, tactic, example)
- An importance score from 0.0 to 1.0

Format each line as:
📝 content | type | importance

Extract aggressively. Every specific technique, named framework, quoted statistic, referenced book/tool/person, mental model, or tactical step should be its own 📝 line. Break compound ideas into separate lines. Don't summarize — atomize."""


class VideoService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.memory_service = MemoryService(db)
        self.llm = get_chat_llm()

    async def ingest(self, user_id: str, url: str) -> dict:
        logger.info(f"Ingesting video for user {user_id}: {url}")

        try:
            result = subprocess.run(
                ["yt-dlp", "--dump-json", "--skip-download", "--no-warnings",
                 "--extractor-args", "youtube:skip=webpage;player_client=android",
                 url],
                capture_output=True, text=True, timeout=60,
            )
        except FileNotFoundError:
            raise VideoIngestionError("yt-dlp not installed")
        if result.returncode != 0:
            raise VideoIngestionError(f"yt-dlp failed: {result.stderr.strip()}")
        meta = json.loads(result.stdout)

        title = meta.get("title", "Unknown")
        channel = meta.get("channel", meta.get("uploader", None))
        duration = meta.get("duration", None)
        description = meta.get("description", "")

        parts = [
            f"Title: {title}",
            *([f"Channel: {channel}"] if channel else []),
            *([f"Duration: {duration}s"] if duration else []),
            f"Description:\n{description[:2000]}",
            "(No subtitles available)",
        ]
        prompt_text = "\n\n".join(parts)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ]
        response = await self.llm.ainvoke(messages)
        raw = response.content
        if isinstance(raw, list):
            output = " ".join(
                b.get("text", str(b)) if isinstance(b, dict) else str(b)
                for b in raw
            )
        else:
            output = str(raw)

        memories = []
        for line in output.splitlines():
            line = line.strip()
            if not line.startswith("📝"):
                continue
            parts = line.removeprefix("📝 ").split("|")
            if len(parts) < 2:
                continue
            content = parts[0].strip()
            memory_type = parts[1].strip()
            try:
                importance = max(0.0, min(1.0, float(parts[2].strip())))
            except (ValueError, IndexError):
                importance = 0.5

            memory = await self.memory_service.add_memory(
                user_id=user_id,
                content=f"[{title}] {content}",
                memory_type=memory_type,
                metadata={"source": "video", "video_title": title, "video_url": url, "channel": channel},
                importance=importance,
            )
            memories.append({
                "id": memory.id,
                "content": memory.content,
                "memory_type": memory.memory_type,
                "importance": memory.importance,
            })

        return {
            "video_title": title,
            "channel": channel,
            "duration": duration,
            "subtitles_available": False,
            "memories": memories,
        }
