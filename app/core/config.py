# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    POSTGRES_DB: str
    REDIS_HOST: str
    REDIS_PORT: int
    CHROMADB_HOST: str
    CHROMADB_PORT: int
    GEMINI_API_KEY: str

    DATABASE_ECHO_SQL: bool = False
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_POOL_TIMEOUT: int = 30

    SECRET_KEY: str = "your_super_secret_key_change_me_in_production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # LLM Context Window & Token Management
    LLM_CONTEXT_WINDOW: int = 30000 # Using a safer default from previous discussion (32k token model)
    LLM_TOKEN_THRESHOLD_PERCENTAGE: float = 0.90
    LLM_RESERVED_TOKENS_OUTPUT: int = 2000 # This is a new setting you added, good for future use

    # Consistent ChromaDB RAG/Memory Settings - Aligned with chat_service.py
    
    # Past Questions & Answers Memory (Layer 1)
    CHROMA_PAST_QA_COLLECTION_NAME: str = "user_past_questions_answers"
    CHROMA_PAST_QA_N_RESULTS: int = 3 # Re-added based on chat_service.py usage
    CHROMA_PAST_QA_SIMILARITY_THRESHOLD: float = 0.1 # Adjusted from 1.0, 1.0 is too strict for similarity

    # General Knowledge Base (Layer 2)
    CHROMA_GENERAL_N_RESULTS: int = 5 # Re-added based on chat_service.py usage
    CHROMA_GENERAL_SIMILARITY_THRESHOLD: float = 0.1 # Re-added based on chat_service.py usage

    # Historical Context (Long-Term Memory - Layer 3)
    CHROMA_HISTORY_COLLECTION_NAME: str = "user_long_term_memory"
    CHROMA_HISTORY_N_RESULTS: int = 2 # Re-added based on chat_service.py usage
    CHROMA_HISTORY_SIMILARITY_THRESHOLD: float = 0.1 # Re-added based on chat_service.py usage

    # Cognitive Knowledge Base (Layer 4)
    CHROMA_COGNITIVE_COLLECTION_NAME: str = "cognitive_knowledge_base"
    CHROMA_COGNITIVE_N_RESULTS: int = 3 # Re-added based on chat_service.py usage
    CHROMA_COGNITIVE_SIMILARITY_THRESHOLD: float = 0.1 # Re-added based on chat_service.py usage

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()