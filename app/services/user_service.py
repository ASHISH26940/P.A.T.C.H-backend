# app/services/user_service.py

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.core.database import User as DBUser # Alias to avoid conflict with Pydantic User
from app.models.user import UserCreate, UserInDB
from app.core.security import get_password # This will be created in Section 3
from typing import Optional

class UserService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_by_username(self, username: str) -> Optional[DBUser]:
        """Retrieves a user by their username."""
        result = await self.db.execute(select(DBUser).where(DBUser.username == username))
        return result.scalar_one_or_none()

    async def create_user(self, user: UserCreate) -> DBUser:
        """Creates a new user in the database."""
        hashed_password = get_password(user.password)
        db_user = DBUser(
            username=user.username,
            email=user.email,
            hashed_password=hashed_password
        )
        self.db.add(db_user)
        await self.db.commit()
        await self.db.refresh(db_user) # Refresh to get auto-generated ID
        return db_user