from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
import bcrypt
from fastapi import HTTPException, status, Depends # Ensure Depends is imported
from fastapi.security import OAuth2PasswordBearer
from app.core.config import settings
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.models.token import TokenData # Ensure this is imported
from app.core.database import get_db, User as DBUser # Ensure these are imported (DBUser is the SQLAlchemy model)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def get_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

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
    
    logger.debug(f"get_current_user called. Token (first 20 chars): {token[:20] if token else 'None'}...")
    
    credentials_exception=HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={
            "WWW-Authenticate":"Bearer"
        }
    )

    try:
        payload=verify_token(token=token)
        logger.debug(f"Token verified successfully. Payload: {payload}")
        
        username:str=payload.get("sub")
        if username is None:
            logger.error("Token payload missing 'sub' claim")
            raise credentials_exception
        
        token_data=TokenData(username=username)
        logger.debug(f"Looking up user in database: {username}")
        
    except JWTError as e:
        logger.error(f"JWT verification failed: {e}")
        raise credentials_exception
    
    user_service=UserService(db=db)
    user =await user_service.get_user_by_username(token_data.username)
    
    if user is None:
        logger.error(f"User not found in database: {token_data.username}")
        raise credentials_exception
    if not user.is_active:
        logger.error(f"Inactive user attempted access: {token_data.username}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="Inactive User")
    
    logger.debug(f"Successfully authenticated user: {username}")
    return user