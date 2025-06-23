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

    SECRET_KEY: str = "your_super_secret_key_change_me_in_production"
    ALGORITHM: str = "HS256" # HS256 is a common symmetric algorithm for JWT
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440


    # LLM Context Window & Token Management
    # Approximate context window for Gemini 1.5 Flash
    LLM_CONTEXT_WINDOW: int = 120000
    # Percentage of context window to use for history before truncation
    LLM_TOKEN_THRESHOLD_PERCENTAGE: float = 0.90
    # Tokens reserved for current user query and AI's potential response
    LLM_RESERVED_TOKENS_OUTPUT: int = 2000

    # General RAG Query Settings
    CHROMA_QUERY_N_RESULTS: int = 4 # Number of documents to retrieve for general RAG

    # Historical Context RAG Settings
    CHROMA_HISTORY_COLLECTION_NAME: str = "user_long_term_memory" # Name of collection for historical context
    CHROMA_HISTORY_QUERY_N_RESULTS: int = 2 # Number of historical documents to retrieve

    # Cognitive Database RAG Settings
    CHROMA_COGNITIVE_COLLECTION_NAME: str = "cognitive_knowledge_base" # Name of collection for cognitive knowledge
    CHROMA_COGNITIVE_QUERY_N_RESULTS: int = 3 # Number of cognitive documents to retrieve

    # Past Questions & Answers Memory Settings
    CHROMA_PAST_QA_COLLECTION_NAME: str = "user_past_questions_answers" # Name of collection for Q&A memory
    # Similarity threshold for considering a question "previously asked" (0.0 to 1.0)
    CHROMA_PAST_QA_SIMILARITY_THRESHOLD: float = 1.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()