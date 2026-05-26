import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy import text

from app.core.database import Base, get_db, User
from app.core.config import settings
from app.core.security import get_password, create_access_token
from app.main import app


TEST_DATABASE_URL = (
    settings.TEST_DATABASE_URL
    or "postgresql+asyncpg://user:password@localhost:5432/patched_test"
)


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="session")
async def dbsession(test_engine):
    factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with factory() as session:
        yield session


@pytest_asyncio.fixture(scope="session")
async def test_user(dbsession):
    from sqlalchemy import select
    result = await dbsession.execute(select(User).where(User.username == "testuser"))
    user = result.scalar_one_or_none()
    if not user:
        user = User(
            username="testuser",
            email=None,
            hashed_password=get_password("testpass123"),
            is_active=True,
        )
        dbsession.add(user)
        await dbsession.commit()
    return user


@pytest_asyncio.fixture(scope="session")
async def auth_token(test_user):
    return create_access_token(data={"sub": test_user.username})


@pytest_asyncio.fixture
async def db_session(test_engine):
    factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with factory() as session:
        yield session


@pytest_asyncio.fixture
async def client(db_session, auth_token):
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        ac.headers["Authorization"] = f"Bearer {auth_token}"
        yield ac

    app.dependency_overrides.clear()
