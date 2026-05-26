import json
import pytest
import subprocess
import uuid
from unittest.mock import AsyncMock, patch, MagicMock
from app.models.memory import Memory


FAKE_JSON = json.dumps({
    "title": "Test Video",
    "channel": "TestChannel",
    "duration": 120,
    "description": "A test video about color grading techniques.",
    "id": "test123",
})

FAKE_LLM_OUTPUT = "📝 Color grading starts with proper white balance | technique | 0.9\n"
FAKE_LLM_OUTPUT_2 = (
    "📝 Color grading starts with proper white balance | technique | 0.9\n"
    "📝 Use waveform monitors not just your eyes | tip | 0.7\n"
)


@pytest.mark.asyncio
async def test_ingest_video_success(client):
    fake_process = subprocess.CompletedProcess([], 0, stdout=FAKE_JSON)

    fake_memory = Memory(
        id=str(uuid.uuid4()),
        user_id="testuser",
        content="[Test Video] Color grading starts with proper white balance",
        memory_type="technique",
        importance=0.9,
    )

    fake_llm = MagicMock()
    fake_llm.ainvoke = AsyncMock(return_value=MagicMock(content=FAKE_LLM_OUTPUT))

    with (
        patch("app.services.video_service.subprocess.run", return_value=fake_process),
        patch("app.services.video_service.get_chat_llm", return_value=fake_llm),
        patch("app.services.video_service.MemoryService.add_memory", new_callable=AsyncMock, return_value=fake_memory),
    ):
        resp = await client.post("/v1/video/ingest", json={
            "url": "https://youtube.com/watch?v=test123",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["video_title"] == "Test Video"
    assert data["channel"] == "TestChannel"
    assert data["duration"] == 120
    assert len(data["memories"]) == 1


@pytest.mark.asyncio
async def test_ingest_video_multiple_memories(client):
    fake_process = subprocess.CompletedProcess([], 0, stdout=FAKE_JSON)

    fake_llm = MagicMock()
    fake_llm.ainvoke = AsyncMock(return_value=MagicMock(content=FAKE_LLM_OUTPUT_2))

    memory_a = Memory(
        id=str(uuid.uuid4()), user_id="testuser",
        content="[Test Video] Color grading starts with proper white balance",
        memory_type="technique", importance=0.9,
    )
    memory_b = Memory(
        id=str(uuid.uuid4()), user_id="testuser",
        content="[Test Video] Use waveform monitors not just your eyes",
        memory_type="tip", importance=0.7,
    )

    add_memory_mock = AsyncMock(side_effect=[memory_a, memory_b])

    with (
        patch("app.services.video_service.subprocess.run", return_value=fake_process),
        patch("app.services.video_service.get_chat_llm", return_value=fake_llm),
        patch("app.services.video_service.MemoryService.add_memory", new_callable=AsyncMock, side_effect=[memory_a, memory_b]),
    ):
        resp = await client.post("/v1/video/ingest", json={
            "url": "https://youtube.com/watch?v=test123",
        })

    assert resp.status_code == 200
    assert len(resp.json()["memories"]) == 2


@pytest.mark.asyncio
async def test_ingest_video_ytdlp_fails(client):
    fake_process = subprocess.CompletedProcess([], 1, stderr="Invalid URL")

    with patch("app.services.video_service.subprocess.run", return_value=fake_process):
        resp = await client.post("/v1/video/ingest", json={
            "url": "https://invalid.url/watch?v=bad",
        })

    assert resp.status_code == 400
    assert "yt-dlp failed" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_ingest_video_empty_llm_output(client):
    fake_process = subprocess.CompletedProcess([], 0, stdout=FAKE_JSON)

    fake_llm = MagicMock()
    fake_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Just some random text without markers"))

    with (
        patch("app.services.video_service.subprocess.run", return_value=fake_process),
        patch("app.services.video_service.get_chat_llm", return_value=fake_llm),
    ):
        resp = await client.post("/v1/video/ingest", json={
            "url": "https://youtube.com/watch?v=test123",
        })

    assert resp.status_code == 200
    assert resp.json()["memories"] == []


@pytest.mark.asyncio
async def test_ingest_video_missing_metadata(client):
    """yt-dlp returns JSON with missing optional fields."""
    minimal_json = json.dumps({"title": "Minimal"})
    fake_process = subprocess.CompletedProcess([], 0, stdout=minimal_json)

    fake_llm = MagicMock()
    fake_llm.ainvoke = AsyncMock(return_value=MagicMock(content=""))

    with (
        patch("app.services.video_service.subprocess.run", return_value=fake_process),
        patch("app.services.video_service.get_chat_llm", return_value=fake_llm),
    ):
        resp = await client.post("/v1/video/ingest", json={
            "url": "https://youtube.com/watch?v=minimal",
        })

    assert resp.status_code == 200
    assert resp.json()["video_title"] == "Minimal"
    assert resp.json()["channel"] is None
    assert resp.json()["duration"] is None
