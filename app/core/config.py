# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    POSTGRES_DB: str # <-- Make sure this is defined here and reads from env
    REDIS_HOST: str
    REDIS_PORT: int
    CHROMADB_HOST: str
    CHROMADB_PORT: int
    GEMINI_API_KEY: str

    DATABASE_ECHO_SQL: bool = False
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_POOL_TIMEOUT: int = 30

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()