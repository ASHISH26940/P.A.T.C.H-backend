from typing import Dict,Any,Annotated
from pydantic import BaseModel,Field

class DynamicContext(BaseModel):
    context_data:Dict[str,Any]=Field(...,description="A dictionary of dynamic context information")


class ContextResponse(BaseModel):
    user_id:str
    context_data:Dict[str,Any]
    updated_at:str