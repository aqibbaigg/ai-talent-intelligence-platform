"""
app/models/job.py
------------------
SQLAlchemy ORM model for the `jobs` table.

Stores job descriptions with a 384-dim embedding so we can
search for matching candidates using FAISS or pgvector.
"""

import uuid
from datetime import datetime
from sqlalchemy import String, Text, DateTime, JSON, Integer, Float
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import ARRAY
from app.core.database import Base


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # ── Job details ───────────────────────────────────────────────
    title:              Mapped[str]       = mapped_column(String(255), nullable=False)
    company:            Mapped[str | None]= mapped_column(String(255), nullable=True)
    description:        Mapped[str]       = mapped_column(Text, nullable=False)
    required_skills:    Mapped[list]      = mapped_column(JSON, default=list)
    nice_to_have_skills:Mapped[list]      = mapped_column(JSON, default=list)
    experience_min:     Mapped[int | None]= mapped_column(Integer, nullable=True)
    experience_max:     Mapped[int | None]= mapped_column(Integer, nullable=True)
    education:          Mapped[str | None]= mapped_column(String(255), nullable=True)
    location:           Mapped[str | None]= mapped_column(String(255), nullable=True)
    job_type:           Mapped[str | None]= mapped_column(String(50), nullable=True)  # full-time, contract

    # ── Embedding ─────────────────────────────────────────────────
    embedding: Mapped[list[float] | None] = mapped_column(
        ARRAY(Float), nullable=True
    )

    # ── Metadata ──────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def __repr__(self) -> str:
        return f"<Job id={self.id} title={self.title}>"
