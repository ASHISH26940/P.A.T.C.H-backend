from typing import TypeAlias
from pydantic import Field,BeforeValidator
from typing_extensions import Annotated

UserID:TypeAlias=Annotated[str,Field(...,min_length=1,max_length=255,description="Unique Idetifier for user")]