"""
app/models/match.py
--------------------
Stores the result of matching a candidate against a job.
Keeps a full breakdown of the weighted score.
"""

import uuid
from datetime import datetime
from sqlalchemy import String, Float, Integer, DateTime, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # ── Foreign keys ──────────────────────────────────────────────
    candidate_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("candidates.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    # ── Scores ────────────────────────────────────────────────────
    final_score:      Mapped[float]      = mapped_column(Float, nullable=False)
    rank:             Mapped[int]        = mapped_column(Integer, nullable=False)
    skill_score:      Mapped[float]      = mapped_column(Float, default=0.0)
    experience_score: Mapped[float]      = mapped_column(Float, default=0.0)
    education_score:  Mapped[float]      = mapped_column(Float, default=0.0)
    cert_score:       Mapped[float]      = mapped_column(Float, default=0.0)
    semantic_score:   Mapped[float]      = mapped_column(Float, default=0.0)

    # ── LLM summary (Week 3) ──────────────────────────────────────
    llm_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return (f"<Match candidate={self.candidate_id[:8]} "
                f"job={self.job_id[:8]} score={self.final_score:.1f}>")
