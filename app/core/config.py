from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    POSTGRES_DB: str
    TEST_DATABASE_URL: str | None = None
    GEMINI_API_KEY: str

    DATABASE_ECHO_SQL: bool = False
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_POOL_TIMEOUT: int = 30

    SECRET_KEY: str = "your_super_secret_key_change_me_in_production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    LLM_CONTEXT_WINDOW: int = 30000
    LLM_TOKEN_THRESHOLD_PERCENTAGE: float = 0.90
    LLM_RESERVED_TOKENS_OUTPUT: int = 2000

    MEMORY_N_RESULTS: int = 5
    MEMORY_SIMILARITY_THRESHOLD: float = 0.3
    MEMORY_IMPORTANCE_DECAY_DAYS: int = 30

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()