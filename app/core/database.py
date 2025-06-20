from sqlalchemy.ext.asyncio import create_async_engine,async_sessionmaker,AsyncSession
from sqlalchemy.orm import declarative_base
from sqlalchemy import text
from app.core.config import settings
from loguru import logger
import asyncio
from typing import AsyncGenerator

Base=declarative_base()

async_engine=None
AsyncSessionLocal=None

async def init_db():
    """
    Initializes the database engine and creates tables if they don't exist.
    This also handles creating the database itself if it doesn't exist,
    which is useful for local development setup.
    """
    global async_engine,AsyncSessionLocal
    if async_engine is not None:
        logger.info("Database engine already initialized.")
        return
    
    server_url = settings.DATABASE_URL

    max_retries=10
    retry_delay=5

    for i in range(max_retries):
        try:
            temp_engine=create_async_engine(server_url,echo=False,pool_pre_ping=True)
            async with temp_engine.connect() as conn:
                result = await conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname='{settings.POSTGRES_DB}'"))
                db_exists = result.scalar_one_or_none()

                if not db_exists:
                    logger.info(f"Database '{settings.POSTGRES_DB}' does not exist. Creating it...")
                    await conn.execute(
                        text(f"CREATE DATABASE {settings.POSTGRES_DB}"),
                        execution_options={"isolation_level": "AUTOCOMMIT"}
                    )
                    logger.info(f"Database '{settings.POSTGRES_DB}' created.")
                else:
                    logger.info(f"Database '{settings.POSTGRES_DB}' already exists.")
            await temp_engine.dispose()

            async_engine = create_async_engine(
                server_url,
                echo=settings.DATABASE_ECHO_SQL,
                pool_size=settings.DATABASE_POOL_SIZE,
                max_overflow=settings.DATABASE_MAX_OVERFLOW,
                pool_timeout=settings.DATABASE_POOL_TIMEOUT,
                pool_pre_ping=True # Ensures connections are alive
            )

            AsyncSessionLocal = async_sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=async_engine,
                class_=AsyncSession,
                expire_on_commit=False # Prevents objects from expiring after commit
            )

            async with async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            logger.info("Database tables initialized successfully (or already existed).")
            break # Exit loop if successful

        except Exception as e:
            logger.error(f"Failed to connect to database or create tables (Attempt {i+1}/{max_retries}): {e}")
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