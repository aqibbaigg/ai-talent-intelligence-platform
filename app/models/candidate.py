"""
app/models/candidate.py
------------------------
SQLAlchemy ORM model for the `candidates` table.

The `embedding` column stores 384-dimensional float32 vectors
using pgvector. This enables fast similarity search directly
in PostgreSQL without a separate vector database.
"""

import uuid
from datetime import datetime
from sqlalchemy import String, Text, Float, DateTime, JSON
from sqlalchemy.dialects.postgresql import ARRAY , JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Candidate(Base):
    __tablename__ = "candidates"

    # ── Primary key ───────────────────────────────────────────────
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # ── Personal info ─────────────────────────────────────────────
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # ── Resume content ────────────────────────────────────────────
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # ── Extracted structured data (JSON) ──────────────────────────
    skills: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="List of extracted skill strings",
    )
    experience_years: Mapped[float | None] = mapped_column(Float, nullable=True)
    experience_raw: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="Raw experience text e.g. '3 years at Google'",
    )
    education: Mapped[str | None] = mapped_column(String(500), nullable=True)
    certifications: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # ── Embedding (384-dim for all-MiniLM-L6-v2) ─────────────────
    embedding: Mapped[list[float] | None] = mapped_column(
        ARRAY(Float),
        nullable=True,
        comment="384-dimensional sentence embedding",
    )

    # ── Metadata ──────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Candidate id={self.id} name={self.name} skills={len(self.skills)}>"
