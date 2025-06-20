import redis.asyncio as redis
from app.core.config import settings
from loguru import logger

_redis_client: redis.Redis = None


async def get_redis_client() -> redis.Redis:
    if _redis_client is None:
        logger.error("Redis client not initialized")
        raise ConnectionError("Redis client not initialized")
    return _redis_client


async def init_redis_client():
    global _redis_client
    try:
        _redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=0,
            decode_responses=False,
        )
        await _redis_client.ping()
        logger.info("Redis client succesfully initialized and connected")
    except Exception as e:
        logger.error(f"Failed to connect to redis :{e}")
        _redis_client = None


async def close_redis_client():
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        logger.info("Redis connection is closed")
