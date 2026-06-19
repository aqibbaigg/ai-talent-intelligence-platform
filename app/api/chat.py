"""
app/api/chat.py
----------------
FastAPI router for all LLM-powered recruiter copilot endpoints.

POST /api/v1/chat                              RAG Q&A over all resumes
POST /api/v1/recommend/{candidate_id}          Candidate fit summary
POST /api/v1/interview-questions/{candidate_id} Tailored interview questions
POST /api/v1/job-summary/{job_id}              AI job description summary
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger

from app.core.database import get_db
from app.models.candidate import Candidate
from app.models.job import Job
from app.schemas.job import ChatRequest, ChatResponse
from app.services.rag_service import answer_recruiter_question
from app.services.llm_service import (
    generate_candidate_summary,
    generate_interview_questions,
    generate_job_summary,
)

router = APIRouter(prefix="/api/v1", tags=["AI Recruiter Copilot"])


# ─────────────────────────────────────────────────────────────────
#  POST /chat  — RAG Q&A over all resumes
# ─────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask AI questions about your candidate pool",
    description="""
RAG-powered Q&A over all uploaded resumes.

Example questions:
- "Who is the best candidate for a Python FastAPI role?"
- "Find candidates with NLP and Docker experience"
- "Which candidates have machine learning and SQL skills?"

Pipeline:
1. Embed your question
2. FAISS semantic search across all candidates
3. Retrieve top-k most relevant profiles
4. Ollama llama3.2 synthesises a recruiter-friendly answer
    """,
)
async def chat(
    request: ChatRequest,
    db:      AsyncSession = Depends(get_db),
) -> ChatResponse:
    try:
        result = await answer_recruiter_question(
            question=request.question,
            db=db,
            top_k=request.top_k,
        )
        return ChatResponse(
            question   = request.question,
            answer     = result["answer"],
            sources    = result["sources"],
            model_used = result["model_used"],
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Chat endpoint failed: {}", e)
        raise HTTPException(status_code=500, detail="Chat request failed")


# ─────────────────────────────────────────────────────────────────
#  POST /recommend/{candidate_id}  — candidate fit summary
# ─────────────────────────────────────────────────────────────────

@router.post(
    "/recommend/{candidate_id}",
    summary="Generate AI recommendation for a candidate",
    description="""
Generate a recruiter-friendly AI summary explaining a candidate's strengths.
Optionally pass job_id to get a role-specific fit assessment.

Returns 3-4 sentences covering:
- Key skills and experience
- Role fit (if job_id provided)
- Overall recommendation
    """,
)
async def recommend(
    candidate_id: str,
    job_id:       str | None = Query(default=None, description="Optional job ID for role-specific fit"),
    db:           AsyncSession = Depends(get_db),
) -> dict:
    # Fetch candidate
    result    = await db.execute(select(Candidate).where(Candidate.id == candidate_id))
    candidate = result.scalar_one_or_none()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # Optionally fetch job
    job = None
    if job_id:
        job_result = await db.execute(select(Job).where(Job.id == job_id))
        job        = job_result.scalar_one_or_none()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

    try:
        summary = await generate_candidate_summary(candidate, job)
        return {
            "candidate_id": candidate_id,
            "name":         candidate.name,
            "job_id":       job_id,
            "summary":      summary,
            "model_used":   "llama3.2",
        }
    except Exception as e:
        logger.error("Recommend endpoint failed: {}", e)
        raise HTTPException(status_code=500, detail="Recommendation generation failed")


# ─────────────────────────────────────────────────────────────────
#  POST /interview-questions/{candidate_id}
# ─────────────────────────────────────────────────────────────────

@router.post(
    "/interview-questions/{candidate_id}",
    summary="Generate tailored interview questions for a candidate",
    description="""
Generate 5 tailored interview questions based on the candidate's skills and experience.
Optionally pass job_id to focus questions on role-specific requirements.

Mix of technical and behavioral questions.
    """,
)
async def interview_questions(
    candidate_id: str,
    job_id:       str | None = Query(default=None, description="Optional job ID for role-focused questions"),
    db:           AsyncSession = Depends(get_db),
) -> dict:
    # Fetch candidate
    result    = await db.execute(select(Candidate).where(Candidate.id == candidate_id))
    candidate = result.scalar_one_or_none()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # Optionally fetch job
    job = None
    if job_id:
        job_result = await db.execute(select(Job).where(Job.id == job_id))
        job        = job_result.scalar_one_or_none()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

    try:
        questions = await generate_interview_questions(candidate, job)
        return {
            "candidate_id": candidate_id,
            "name":         candidate.name,
            "job_id":       job_id,
            "questions":    questions,
            "model_used":   "llama3.2",
        }
    except Exception as e:
        logger.error("Interview questions endpoint failed: {}", e)
        raise HTTPException(status_code=500, detail="Interview question generation failed")


# ─────────────────────────────────────────────────────────────────
#  POST /job-summary/{job_id}
# ─────────────────────────────────────────────────────────────────

@router.post(
    "/job-summary/{job_id}",
    summary="Generate AI summary of a job posting",
    description="""
Let the LLM summarize job requirements in plain English.

Returns:
- summary: 2-3 sentence overview
- must_have_skills: extracted required skills
- nice_to_have_skills: extracted optional skills
    """,
)
async def job_summary(
    job_id: str,
    db:     AsyncSession = Depends(get_db),
) -> dict:
    # Fetch job
    result = await db.execute(select(Job).where(Job.id == job_id))
    job    = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        summary_data = await generate_job_summary(job)
        return {
            "job_id":              job_id,
            "title":               job.title,
            "model_used":          "llama3.2",
            **summary_data,
        }
    except Exception as e:
        logger.error("Job summary endpoint failed: {}", e)
        raise HTTPException(status_code=500, detail="Job summary generation failed")