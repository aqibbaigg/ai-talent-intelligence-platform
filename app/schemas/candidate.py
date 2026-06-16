"""
app/schemas/candidate.py
-------------------------
Pydantic v2 schemas for request validation and response serialisation.

Why separate schemas from models?
  ORM models (SQLAlchemy) define the DB structure.
  Schemas (Pydantic) define the API contract.
  Keeping them separate means you can change the DB without
  breaking the API and vice versa.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, field_validator


# ── Resume upload response ────────────────────────────────────────

class CandidateBase(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None


class CandidateCreate(CandidateBase):
    """Internal schema — used after parsing, before DB insert."""
    raw_text: str
    file_name: str
    skills: list[str] = []
    experience_years: Optional[float] = None
    experience_raw: Optional[str] = None
    education: Optional[str] = None
    certifications: list[str] = []
    embedding: Optional[list[float]] = None


class CandidateResponse(CandidateBase):
    """
    What the API returns after a successful resume upload.
    Never returns the raw embedding vector (too large, not useful to clients).
    """
    id: str
    file_name: str
    skills: list[str]
    experience_years: Optional[float]
    experience_raw: Optional[str]
    education: Optional[str]
    certifications: list[str]
    embedding_generated: bool
    created_at: datetime

    model_config = {"from_attributes": True}   # replaces orm_mode in Pydantic v2

    @field_validator("embedding_generated", mode="before")
    @classmethod
    def check_embedding(cls, v):
        return bool(v)


class CandidateList(BaseModel):
    """Paginated list response."""
    total: int
    candidates: list[CandidateResponse]


# ── Parsed resume (internal, not exposed to API) ──────────────────

class ParsedResume(BaseModel):
    """
    Intermediate object produced by the parser service.
    Passed between services before DB insert.
    """
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    raw_text: str
    skills: list[str] = []
    experience_years: Optional[float] = None
    experience_raw: Optional[str] = None
    education: Optional[str] = None
    certifications: list[str] = []

    model_config = {"from_attributes": True}


# ── Upload error response ─────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
