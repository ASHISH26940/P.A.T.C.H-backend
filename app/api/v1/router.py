from fastapi import APIRouter, Depends
from app.api.v1.endpoints import persona, context, chat, auth
from app.api.v1.endpoints import document as memory_endpoints
from app.api.v1.endpoints import health as health_endpoint
from app.api.v1.endpoints import video as video_endpoints
from app.api.v1.endpoints import user as user_endpoints
from app.api.v1.endpoints import memory_links as memory_links_endpoints
from app.core.security import get_current_user

api_router = APIRouter()

protected = [Depends(get_current_user)]

api_router.include_router(persona.router, prefix="/persona", tags=["Persona"], dependencies=protected)
api_router.include_router(context.router, prefix="/context", tags=["Context"], dependencies=protected)
api_router.include_router(memory_endpoints.router, prefix="/memory", tags=["Memory"], dependencies=protected)
api_router.include_router(memory_links_endpoints.router, prefix="/memory", tags=["Memory"], dependencies=protected)
api_router.include_router(video_endpoints.router, prefix="/video", tags=["Video"], dependencies=protected)
api_router.include_router(user_endpoints.router, prefix="/user", tags=["User"], dependencies=protected)
api_router.include_router(chat.router, prefix="/chat", tags=["Chat"], dependencies=protected)
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(health_endpoint.router, prefix="/health", tags=["Health"])