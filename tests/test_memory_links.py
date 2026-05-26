import pytest
import uuid
from app.models.memory import Memory, MemoryLink


@pytest.mark.asyncio
async def test_create_link(client, db_session):
    m1 = Memory(id=str(uuid.uuid4()), user_id="testuser", content="Source memory", memory_type="test")
    m2 = Memory(id=str(uuid.uuid4()), user_id="testuser", content="Target memory", memory_type="test")
    db_session.add_all([m1, m2])
    await db_session.commit()

    resp = await client.post("/v1/memory/links", json={
        "source_memory_id": m1.id,
        "target_memory_id": m2.id,
        "relationship": "depends_on",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["source_memory_id"] == m1.id
    assert data["target_memory_id"] == m2.id
    assert data["relationship"] == "depends_on"
    assert "id" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_create_link_default_relationship(client, db_session):
    m1 = Memory(id=str(uuid.uuid4()), user_id="testuser", content="A", memory_type="test")
    m2 = Memory(id=str(uuid.uuid4()), user_id="testuser", content="B", memory_type="test")
    db_session.add_all([m1, m2])
    await db_session.commit()

    resp = await client.post("/v1/memory/links", json={
        "source_memory_id": m1.id,
        "target_memory_id": m2.id,
    })
    assert resp.status_code == 201
    assert resp.json()["relationship"] == "related_to"


@pytest.mark.asyncio
async def test_create_link_source_not_found(client):
    resp = await client.post("/v1/memory/links", json={
        "source_memory_id": "00000000-0000-0000-0000-000000000001",
        "target_memory_id": "00000000-0000-0000-0000-000000000002",
    })
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_link_target_not_found(client, db_session):
    m1 = Memory(id=str(uuid.uuid4()), user_id="testuser", content="A", memory_type="test")
    db_session.add(m1)
    await db_session.commit()

    resp = await client.post("/v1/memory/links", json={
        "source_memory_id": m1.id,
        "target_memory_id": "00000000-0000-0000-0000-000000000099",
    })
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_memory_links(client, db_session):
    m1 = Memory(id=str(uuid.uuid4()), user_id="testuser", content="A", memory_type="test")
    m2 = Memory(id=str(uuid.uuid4()), user_id="testuser", content="B", memory_type="test")
    link = MemoryLink(
        id=str(uuid.uuid4()),
        source_memory_id=m1.id,
        target_memory_id=m2.id,
        relationship="references",
    )
    db_session.add_all([m1, m2, link])
    await db_session.commit()

    resp = await client.get(f"/v1/memory/links/{m1.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["links"]) == 1
    assert data["links"][0]["relationship"] == "references"
    assert data["links"][0]["source_memory_id"] == m1.id
    assert data["links"][0]["target_memory_id"] == m2.id


@pytest.mark.asyncio
async def test_get_memory_links_memory_not_found(client):
    resp = await client.get("/v1/memory/links/00000000-0000-0000-0000-000000000099")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_memory_links_empty(client, db_session):
    m1 = Memory(id=str(uuid.uuid4()), user_id="testuser", content="Alone", memory_type="test")
    db_session.add(m1)
    await db_session.commit()

    resp = await client.get(f"/v1/memory/links/{m1.id}")
    assert resp.status_code == 200
    assert resp.json()["links"] == []


@pytest.mark.asyncio
async def test_delete_link(client, db_session):
    m1 = Memory(id=str(uuid.uuid4()), user_id="testuser", content="A", memory_type="test")
    m2 = Memory(id=str(uuid.uuid4()), user_id="testuser", content="B", memory_type="test")
    link = MemoryLink(
        id=str(uuid.uuid4()),
        source_memory_id=m1.id,
        target_memory_id=m2.id,
        relationship="related_to",
    )
    db_session.add_all([m1, m2, link])
    await db_session.commit()

    resp = await client.delete(f"/v1/memory/links/{link.id}")
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_delete_link_not_found(client):
    resp = await client.delete("/v1/memory/links/00000000-0000-0000-0000-000000000099")
    assert resp.status_code == 404
