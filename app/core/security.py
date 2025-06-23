from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends # Ensure Depends is imported
from fastapi.security import OAuth2PasswordBearer
from app.core.config import settings
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.token import TokenData # Ensure this is imported
from app.core.database import get_db, User as DBUser # Ensure these are imported (DBUser is the SQLAlchemy model)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password:str,hashed_password:str):
    return pwd_context.verify(plain_password,hashed_password)


def get_password(password:str)->str:
    return pwd_context.hash(password)

oauth2_scheme=OAuth2PasswordBearer(tokenUrl="/v1/auth/token")

def create_access_token(data:dict,expires_delta:Optional[timedelta]=None)->str:
    to_encode=data.copy()
    if expires_delta:
        expire=datetime.now(timezone.utc)+expires_delta
    else:
        expire=datetime.now(timezone.utc)+timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({
        "exp":expire
    })
    encoded_jwt=jwt.encode(to_encode,settings.SECRET_KEY,algorithm=settings.ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> dict:
    """Verifies and decodes a JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decode the token using your secret key and algorithm
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        # The 'sub' (subject) claim usually holds the user identifier (e.g., username)
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        # You can add more validation here, e.g., token type
        return payload
    except JWTError:
        raise credentials_exception
    

async def get_current_user(
        token:str=Depends(oauth2_scheme),
        db:AsyncSession=Depends(get_db)
)->DBUser:
    """
    Dependency to get the current authenticated user from a JWT token.
    Raises HTTPException if credentials are invalid or user not found.
    """
    from app.services.user_service import UserService
    credentials_exception=HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={
            "WWW-Authenticate":"Bearer"
        }
    )

    try:
        payload=verify_token(token=token)
        username:str=payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data=TokenData(usernmae=username)
    except JWTError:
        raise credentials_exception
    
    user_service=UserService(db=db)
    user =await user_service.get_user_by_username(token_data.usernmae)
    if user is None:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="Inactive User")
    return user