# cogniflow-ai-service/app/models/persona.py
import uuid
from sqlalchemy import Column, String, Text
from sqlalchemy.dialects.postgresql import UUID, ARRAY
# from sqlalchemy import DateTime, func # Uncomment if you add created_at/updated_at timestamps

from app.core.database import Base # Assuming common.py defines your declarative base

class Persona(Base):
    """SQLAlchemy model for the 'personas' table."""
    __tablename__ = "personas"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True) # Text for longer strings
    traits = Column(ARRAY(String), nullable=False, default=[]) # List of strings
    goals = Column(ARRAY(String), nullable=False, default=[])   # List of strings

    # Uncomment if you want automatic timestamps
    # created_at = Column(DateTime(timezone=True), server_default=func.now())
    # updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Persona(id={self.id}, name='{self.name}')>"