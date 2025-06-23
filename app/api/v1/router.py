# app/api/v1/router.py
from fastapi import APIRouter,Depends
from app.api.v1.endpoints import persona, context, document,chat,auth
from app.api.v1.endpoints import health as health_endpoint # <--- Alias the health module
from app.core.security import get_current_user


api_router = APIRouter()

protected_dependencies=[Depends(get_current_user)]

api_router.include_router(
    persona.router,
    prefix="/persona",
    tags=["Persona"],
    dependencies=protected_dependencies # <--- Apply dependency here
)
api_router.include_router(
    context.router,
    prefix="/context",
    tags=["Context"],
    dependencies=protected_dependencies # <--- Apply dependency here
)
api_router.include_router(
    document.router,
    prefix="/documents",
    tags=["Documents"],
    dependencies=protected_dependencies # <--- Apply dependency here
)
api_router.include_router(
    chat.router,
    prefix="/chat",
    tags=["AI Chat"],
    dependencies=protected_dependencies # <--- Apply dependency here
)

# --- Public Routes ---
# Authentication endpoints are explicitly NOT protected
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"]) # No dependencies= here

# Health endpoint is also typically public
api_router.include_router(health_endpoint.router, prefix="/health", tags=["Health"])