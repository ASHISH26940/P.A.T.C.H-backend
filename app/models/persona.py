import uuid
from sqlalchemy import Column, String, Text
from sqlalchemy.dialects.postgresql import UUID, ARRAY

from app.core.database import Base

class Persona(Base):
    __tablename__ = "personas"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(String, nullable=False, index=True)
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    traits = Column(ARRAY(String), nullable=False, default=[])
    goals = Column(ARRAY(String), nullable=False, default=[])

    def __repr__(self):
        return f"<Persona(id={self.id}, name='{self.name}')>"