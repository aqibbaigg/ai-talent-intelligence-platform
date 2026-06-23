"""
app/schemas/job.py
-------------------
Pydantic schemas for Job and Match API contracts.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Job schemas ───────────────────────────────────────────────────

class JobCreate(BaseModel):
    title:               str
    company:             Optional[str]  = None
    description:         str            = Field(..., min_length=50)
    required_skills:     list[str]      = []
    nice_to_have_skills: list[str]      = []
    experience_min:      Optional[int]  = None
    experience_max:      Optional[int]  = None
    education:           Optional[str]  = None
    location:            Optional[str]  = None
    job_type:            Optional[str]  = None


class JobResponse(BaseModel):
    id:                  str
    title:               str
    company:             Optional[str]
    description:         str
    required_skills:     list[str]
    nice_to_have_skills: list[str]
    experience_min:      Optional[int]
    experience_max:      Optional[int]
    education:           Optional[str]
    location:            Optional[str]
    job_type:            Optional[str]
    embedding_generated: bool
    created_at:          datetime

    model_config = {"from_attributes": True}


# ── Match schemas ─────────────────────────────────────────────────

class ScoreBreakdown(BaseModel):
    skill_score:      float
    experience_score: float
    education_score:  float
    cert_score:       float
    semantic_score:   float


class CandidateMatchResult(BaseModel):
    rank:             int
    candidate_id:     str
    name:             str
    email:            Optional[str]
    skills:           list[str]
    experience_years: Optional[float]
    education:        Optional[str]
    final_score:      float
    ats_score:        float = 0.0        # ← NEW
    score_breakdown:  ScoreBreakdown
    llm_summary:      Optional[str] = None


class MatchResponse(BaseModel):
    job_id:                   str
    job_title:                str
    total_candidates_scanned: int
    top_matches:              list[CandidateMatchResult]


# ── RAG / Chat schemas ────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=5)
    top_k:    int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    question:   str
    answer:     str
    sources:    list[dict]
    model_used: str