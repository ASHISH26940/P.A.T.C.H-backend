from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from app.core.config import settings
from loguru import logger


try:
    logger.info(settings.GEMINI_API_KEY)
    chat_llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="AIzaSyDfOPojqsLNxt9-k8F2WAxHN623bt7lFes", temperature=0.7)
    embeddings_models=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=settings.GEMINI_API_KEY)
    logger.info("Google Gemini llm and embedings models initialized")
except Exception as e:
    logger.error(f"Failed to initialize google gemini models: {e}")
    chat_llm=None
    embeddings_models=None


def get_chat_llm():
    if chat_llm is None:
        logger.error("Chat LLM not initialized. Check GEMINI_API_KEY.")
        raise ValueError("Chat LLM not available.")
    return chat_llm

def get_embeddings_model():
    if embeddings_models is None:
        logger.error("Embeddings model not initialized. Check GEMINI_API_KEY.")
        raise ValueError("Embeddings model not available.")
    return embeddings_models

