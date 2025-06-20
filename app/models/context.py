# app/models/context.py
import datetime
from sqlalchemy import Column, String, JSON, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
import uuid

# Import Base from your database configuration
from app.core.database import Base

class Context(Base):
    __tablename__ = "contexts"

    # Corrected: Use 'default' with a callable for ORM-generated defaults,
    # not 'default_factory' unless using MappedAsDataclass.
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    context_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.datetime.now, onupdate=datetime.datetime.now
    )

    def __repr__(self):
        return f"<Context(user_id='{self.user_id}', updated_at='{self.updated_at}')>"