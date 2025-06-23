from fastapi import APIRouter,Depends,HTTPException,status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db,User as DBUser
from app.core.security import verify_password,create_access_token
from app.models.user import UserCreate,User
from app.models.token import Token
from app.core.security import get_current_user

router=APIRouter()

@router.post("/register",status_code=status.HTTP_201_CREATED,summary="Register new User")
async def register_user(
        user_data:UserCreate,
        db:AsyncSession=Depends(get_db)
):
    from app.services.user_service import UserService
    user_service=UserService(db)
    db_user=await user_service.get_user_by_username(user_data.username)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already existed"
        )
    created_user=await user_service.create_user(user_data)
    return created_user

@router.post("/login", response_model=Token, summary="Authenticate user and get JWT access token")
async def login_for_access_token(
    form_data:OAuth2PasswordRequestForm=Depends(),
    db:AsyncSession=Depends(get_db)
):
    from app.services.user_service import UserService
    user_service=UserService(db=db)
    user=await user_service.get_user_by_username(form_data.username)
    if not user or not verify_password(form_data.password,user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate":"Bearer"}
        )

    access_token=create_access_token(
        data={"sub":user.username}
    )
    return {
        "access_token":access_token,
        "token_type":"bearer"
    }


@router.get("/me", response_model=User) # Use your Pydantic User schema as response_model
async def read_current_user(current_user: DBUser = Depends(get_current_user)):
    """
    Retrieve the current authenticated user's information.
    This endpoint requires a valid JWT access token.
    """
    # CRITICAL: Ensure you are returning the current_user object
    return current_user