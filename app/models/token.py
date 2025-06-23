from pydantic import BaseModel,Field

class Token(BaseModel):
    access_token:str=Field(...,description="The JWT access token")
    token_type:str=Field("bearer",description="The type of token (e.g., 'bearer')")

class TokenData(BaseModel):
    usernmae:str|None=None

    