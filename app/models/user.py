# app/models/user.py

from pydantic import BaseModel, EmailStr, Field
from typing import Optional

# Model for user creation requests (e.g., /register)
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    password: str = Field(..., min_length=8) # Passwords will be hashed

# Model for a user as stored in the database (including hashed password)
class UserInDB(UserCreate):
    id: Optional[int] = None
    hashed_password: str

    class Config:
        from_attributes = True # For Pydantic v2+, enables mapping from ORM objects

# Model for current authenticated user (e.g., what endpoints receive after auth)
class User(BaseModel):
    id: int
    username: str
    email: Optional[EmailStr] = None

    class Config:
        from_attributes = True