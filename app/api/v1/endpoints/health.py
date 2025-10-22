# app/api/v1/endpoints/health.py
from fastapi import APIRouter, Response
from loguru import logger

router = APIRouter()

@router.get("/") # <--- CHANGE THIS LINE from "/health" to "/"
async def health():
    return Response("Hello world", media_type="text/plain", status_code=200)
    # Or: return PlainTextResponse("Hello world")