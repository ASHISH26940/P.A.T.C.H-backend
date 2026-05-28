from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.core.database import get_db, User
from app.core.security import get_current_user

router = APIRouter()


class CookiesUpdate(BaseModel):
    cookies: str


class CookiesResponse(BaseModel):
    cookies: str | None


@router.put("/cookies", response_model=CookiesResponse)
async def update_cookies(
    body: CookiesUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user = await db.get(User, current_user.id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.youtube_cookies = body.cookies
    await db.commit()
    await db.refresh(user)
    return CookiesResponse(cookies=user.youtube_cookies)


@router.get("/cookies", response_model=CookiesResponse)
async def get_cookies(
    current_user: User = Depends(get_current_user),
):
    return CookiesResponse(cookies=current_user.youtube_cookies)
