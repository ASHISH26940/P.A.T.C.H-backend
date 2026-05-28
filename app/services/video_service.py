import subprocess
import json
import tempfile
import os
import httpx
import re
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.config import settings
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


def _extract_video_id(url: str) -> str | None:
    m = re.search(r"(?:v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/|youtube\.com/v/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


async def _fetch_via_youtube_api(url: str) -> dict:
    """Fallback: fetch video metadata via YouTube Data API v3."""
    video_id = _extract_video_id(url)
    if not video_id:
        raise VideoIngestionError("Could not extract video ID from URL")
    if not settings.YOUTUBE_API_KEY:
        raise VideoIngestionError("YouTube API key not configured")

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={
                "part": "snippet,contentDetails",
                "id": video_id,
                "key": settings.YOUTUBE_API_KEY,
            },
        )
    if resp.status_code != 200:
        raise VideoIngestionError(f"YouTube API returned {resp.status_code}: {resp.text}")

    data = resp.json()
    items = data.get("items", [])
    if not items:
        raise VideoIngestionError("Video not found on YouTube")

    snippet = items[0]["snippet"]
    duration_iso = items[0]["contentDetails"]["duration"]
    import isodate
    duration_seconds = int(isodate.parse_duration(duration_iso).total_seconds())

    return {
        "video_title": snippet["title"],
        "channel": snippet["channelTitle"],
        "duration": duration_seconds,
        "subtitles_available": False,
        "description": snippet.get("description", ""),
    }


class VideoService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.memory_service = MemoryService(db)
        self.llm = get_chat_llm()

    async def ingest(self, user_id: str, url: str) -> dict:
        logger.info(f"Ingesting video for user {user_id}: {url}")

        meta = await self._fetch_metadata(url)

        title = meta.get("title", "Unknown")
        channel = meta.get("channel", None)
        duration = meta.get("duration", None)
        description = meta.get("description", "")

        memories = await self._extract_memories(user_id, url, title, channel, description)

        return {
            "video_title": title,
            "channel": channel,
            "duration": duration,
            "subtitles_available": False,
            "memories": memories,
        }

    async def _fetch_metadata(self, url: str) -> dict:
        """Try yt-dlp first, fall back to YouTube Data API."""
        meta = await self._try_ytdlp(url)
        if meta:
            return meta
        logger.warning("yt-dlp failed, trying YouTube API fallback")
        api_meta = await _fetch_via_youtube_api(url)
        return {
            "title": api_meta["video_title"],
            "channel": api_meta["channel"],
            "duration": api_meta["duration"],
            "description": api_meta.get("description", ""),
        }

    async def _try_ytdlp(self, url: str) -> dict | None:
        cmd = [
            "yt-dlp", "--dump-json", "--skip-download", "--no-warnings",
            "--extractor-args", "youtube:player_client=tv,ios,android;skip=webpage",
            "--user-agent", "Mozilla/5.0 (Linux; Android 14; Pixel 8 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.113 Mobile Safari/537.36",
            url,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except FileNotFoundError:
            logger.warning("yt-dlp not installed")
            return None
        if result.returncode != 0:
            logger.warning(f"yt-dlp failed: {result.stderr.strip()}")
            return None
        try:
            meta = json.loads(result.stdout)
            return {
                "title": meta.get("title", "Unknown"),
                "channel": meta.get("channel", meta.get("uploader", None)),
                "duration": meta.get("duration", None),
                "description": meta.get("description", ""),
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"yt-dlp parse error: {e}")
            return None

    async def _extract_memories(self, user_id: str, url: str, title: str, channel: str | None, description: str) -> list[dict]:
        parts = [
            f"Title: {title}",
            *([f"Channel: {channel}"] if channel else []),
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

        return memories
