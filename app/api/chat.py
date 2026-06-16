"""
app/api/chat.py
----------------
FastAPI router for LLM-powered RAG endpoints.

POST /api/v1/chat              RAG: answer recruiter questions
POST /api/v1/recommend         LLM: why hire this candidate?
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.database import get_db
from app.schemas.job import ChatRequest, ChatResponse
from app.services.rag_service import (
    answer_recruiter_question,
    generate_candidate_recommendation,
)

router = APIRouter(prefix="/api/v1", tags=["AI chat & recommendations"])


# ─────────────────────────────────────────────────────────────────
#  POST /chat  — RAG over all resumes
# ─────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask AI questions about your candidate pool",
    description="""
    RAG-powered Q&A over all uploaded resumes.

    Example questions:
    - "Find candidates experienced in NLP and AWS"
    - "Who has worked at a startup and knows React and TypeScript?"
    - "List all candidates with 5+ years Python experience"
    - "Which candidates have machine learning and SQL skills?"

    Pipeline:
    1. Embed your question using sentence-transformers
    2. FAISS semantic search across all candidate embeddings
    3. Retrieve top-k most relevant candidate profiles
    4. GPT-4o synthesises a recruiter-friendly answer

    Requires OPENAI_API_KEY in .env for LLM step.
    Falls back to template answer if key not set.
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
#  POST /recommend  — why hire this candidate?
# ─────────────────────────────────────────────────────────────────

@router.post(
    "/recommend",
    summary="Generate LLM recommendation for a candidate",
    description="""
    Generate a recruiter-friendly AI summary explaining why a specific
    candidate is a good (or poor) fit for a job.

    Returns 3-4 sentences mentioning:
    - Skill alignment
    - Experience fit
    - Overall recommendation

    Requires a prior call to POST /match so scores exist in DB.
    """,
)
async def recommend(
    candidate_id: str,
    job_id:       str,
    db:           AsyncSession = Depends(get_db),
) -> dict:
    try:
        summary = await generate_candidate_recommendation(
            candidate_id=candidate_id,
            job_id=job_id,
            db=db,
        )
        return {
            "candidate_id": candidate_id,
            "job_id":       job_id,
            "summary":      summary,
        }
    except Exception as e:
        logger.error("Recommendation failed: {}", e)
        raise HTTPException(status_code=500, detail="Recommendation generation failed")
