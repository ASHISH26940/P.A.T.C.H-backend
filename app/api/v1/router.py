# app/api/v1/router.py
from fastapi import APIRouter
from app.api.v1.endpoints import persona, context, document,chat
from app.api.v1.endpoints import health as health_endpoint # <--- Alias the health module

api_router = APIRouter()

api_router.include_router(persona.router, prefix="/persona", tags=["Persona"])
api_router.include_router(context.router, prefix="/context", tags=["Context"])
api_router.include_router(document.router, prefix="/documents", tags=["Documents"])
api_router.include_router(chat.router,prefix="/chat",tags=["chats"])

# Correct way to include the router from the health endpoint module
api_router.include_router(health_endpoint.router, prefix="/health", tags=["Health"])
# ^^^^^^^^^^^^^^^^^^^^^^^  Note: health_endpoint.router