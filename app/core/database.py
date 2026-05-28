from sqlalchemy.ext.asyncio import create_async_engine,async_sessionmaker,AsyncSession
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import text, Column, Integer, String, Boolean, Text
from app.core.config import settings
from loguru import logger
import asyncio
from typing import AsyncGenerator, Optional

Base=declarative_base()

async_engine=None
AsyncSessionLocal=None

async def init_db():
    global async_engine,AsyncSessionLocal
    if async_engine is not None:
        logger.info("Database engine already initialized.")
        return
    
    raw_url = settings.DATABASE_URL
    if raw_url.startswith("postgresql://") and "+asyncpg" not in raw_url:
        raw_url = raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    server_url = raw_url

    max_retries=10
    retry_delay=5

    for i in range(max_retries):
        try:
            async_engine = create_async_engine(
                server_url,
                echo=settings.DATABASE_ECHO_SQL,
                pool_size=settings.DATABASE_POOL_SIZE,
                max_overflow=settings.DATABASE_MAX_OVERFLOW,
                pool_timeout=settings.DATABASE_POOL_TIMEOUT,
                pool_pre_ping=True
            )

            async with async_engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.run_sync(Base.metadata.create_all)

            AsyncSessionLocal = async_sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            logger.info("Database initialized with pgvector extension and tables.")
            break

        except Exception as e:
            logger.error(f"Failed to connect to database (Attempt {i+1}/{max_retries}): {e}")
            if i < max_retries - 1:
                logger.info(f"Retrying database connection in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.critical("Maximum database connection retries reached. Exiting startup.")
                raise

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides an asynchronous database session.
    It ensures the session is closed after the request.
    """
    if AsyncSessionLocal is None:
        # This case should ideally not happen if init_db is called on startup
        logger.error("AsyncSessionLocal is not initialized. Calling init_db...")
        await init_db() # Attempt to initialize if not already
        if AsyncSessionLocal is None: # If init_db still fails
            raise RuntimeError("Database session local could not be initialized.")

    db = AsyncSessionLocal()
    try:
        yield db
    finally:
        await db.close()


async def dispose_db():
    """Disposes the database engine connections."""
    global async_engine
    if async_engine:
        await async_engine.dispose()
        logger.info("Database engine connections disposed.")


class User(Base):
    __tablename__="users"

    id:Mapped[int]=mapped_column(Integer,primary_key=True,index=True)
    username:Mapped[str]=mapped_column(String,unique=True,index=True)
    email:Mapped[Optional[str]]=mapped_column(String,unique=True,index=True,nullable=True)
    hashed_password:Mapped[str]=mapped_column(String)
    is_active:Mapped[bool]=mapped_column(Boolean,default=True)
    youtube_cookies:Mapped[Optional[str]]=mapped_column(Text,nullable=True)

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"