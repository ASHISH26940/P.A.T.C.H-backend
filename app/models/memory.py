import uuid
import datetime
from sqlalchemy import Column, String, Text, Float, Integer, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from app.core.database import Base


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    session_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc), index=True
    )
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, user='{self.user_id}', role='{self.role}')>"


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    memory_type: Mapped[str] = mapped_column(
        String(32), index=True, nullable=False, default="general"
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = Column(Vector(3072), nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc), index=True
    )
    last_accessed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self):
        return f"<Memory(id={self.id}, user='{self.user_id}', type='{self.memory_type}')>"


class Extraction(Base):
    __tablename__ = "extractions"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    session_id: Mapped[str | None] = mapped_column(String, nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="chat")
    extracted_insights: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    def __repr__(self):
        return f"<Extraction(id={self.id}, user='{self.user_id}', source='{self.source}')>"


class MemoryLink(Base):
    __tablename__ = "memory_links"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_memory_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    target_memory_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    relationship: Mapped[str] = mapped_column(String(32), nullable=False, default="related_to")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    def __repr__(self):
        return f"<MemoryLink({self.source_memory_id} -[{self.relationship}]-> {self.target_memory_id})>"


class UserContext(Base):
    __tablename__ = "user_contexts"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    context_data: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc), onupdate=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    def __repr__(self):
        return f"<UserContext(user_id='{self.user_id}')>"
