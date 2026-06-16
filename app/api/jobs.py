"""
app/api/jobs.py
----------------
FastAPI router for job and matching endpoints.

POST /api/v1/jobs          Create job posting + embed description
GET  /api/v1/jobs          List all jobs
GET  /api/v1/jobs/{id}     Get single job
POST /api/v1/match         FAISS match candidates to a job
GET  /api/v1/matches/{id}  Get saved match results for a job
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger

from app.core.database import get_db
from app.models.job import Job
from app.models.match import Match
from app.models.candidate import Candidate
from app.schemas.job import (
    JobCreate, JobResponse, MatchResponse, CandidateMatchResult, ScoreBreakdown
)
from app.services import job_service
from app.services.faiss_index import get_faiss_index

router = APIRouter(prefix="/api/v1", tags=["jobs & matching"])


# ─────────────────────────────────────────────────────────────────
#  POST /jobs  — create a new job posting
# ─────────────────────────────────────────────────────────────────

@router.post(
    "/jobs",
    response_model=JobResponse,
    status_code=201,
    summary="Create a job posting",
    description="""
    Create a new job posting. This endpoint:
    1. Accepts job title, description, required skills, experience range
    2. Generates a 384-dim sentence embedding from the job description
    3. Stores everything in PostgreSQL

    After creating a job, call POST /match to find matching candidates.
    """,
)
async def create_job(
    data: JobCreate,
    db:   AsyncSession = Depends(get_db),
) -> JobResponse:
    try:
        job = await job_service.create_job(data, db)
    except Exception as e:
        logger.error("Job creation failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))

    return JobResponse(
        id                  = job.id,
        title               = job.title,
        company             = job.company,
        description         = job.description,
        required_skills     = job.required_skills,
        nice_to_have_skills = job.nice_to_have_skills,
        experience_min      = job.experience_min,
        experience_max      = job.experience_max,
        education           = job.education,
        location            = job.location,
        job_type            = job.job_type,
        embedding_generated = job.embedding is not None,
        created_at          = job.created_at,
    )


# ─────────────────────────────────────────────────────────────────
#  GET /jobs  — list all jobs
# ─────────────────────────────────────────────────────────────────

@router.get("/jobs", response_model=list[JobResponse], summary="List all jobs")
async def list_jobs(
    skip:  int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db:    AsyncSession = Depends(get_db),
) -> list[JobResponse]:
    result = await db.execute(
        select(Job).order_by(Job.created_at.desc()).offset(skip).limit(limit)
    )
    jobs = result.scalars().all()
    return [
        JobResponse(
            id=j.id, title=j.title, company=j.company,
            description=j.description, required_skills=j.required_skills,
            nice_to_have_skills=j.nice_to_have_skills,
            experience_min=j.experience_min, experience_max=j.experience_max,
            education=j.education, location=j.location, job_type=j.job_type,
            embedding_generated=j.embedding is not None, created_at=j.created_at,
        )
        for j in jobs
    ]


# ─────────────────────────────────────────────────────────────────
#  POST /match  — run FAISS matching for a job
# ─────────────────────────────────────────────────────────────────

@router.post(
    "/match",
    response_model=MatchResponse,
    summary="Match candidates to a job using FAISS",
    description="""
    Find the best-matching candidates for a job posting.

    Pipeline:
    1. Fetch job embedding from DB
    2. Search FAISS index for semantically similar candidate embeddings
    3. Fetch candidates from DB
    4. Apply weighted scoring:
       - 50% Skill match (keyword + semantic)
       - 20% Experience
       - 20% Education
       - 10% Certifications
    5. Re-rank by final score
    6. Save matches to DB
    7. Return ranked candidate list with score breakdowns

    Returns top 10 candidates by default.
    """,
)
async def match_candidates(
    job_id: str,
    top_k:  int = Query(default=10, ge=1, le=50),
    db:     AsyncSession = Depends(get_db),
) -> MatchResponse:
    try:
        result = await job_service.match_candidates(job_id, db, top_k)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Matching failed: {}", e)
        raise HTTPException(status_code=500, detail="Matching failed")


# ─────────────────────────────────────────────────────────────────
#  GET /matches/{job_id}  — retrieve saved match results
# ─────────────────────────────────────────────────────────────────

@router.get(
    "/matches/{job_id}",
    response_model=MatchResponse,
    summary="Get saved match results for a job",
)
async def get_matches(
    job_id: str,
    db:     AsyncSession = Depends(get_db),
) -> MatchResponse:
    # Fetch job
    job_result = await db.execute(select(Job).where(Job.id == job_id))
    job = job_result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Fetch saved matches ordered by rank
    match_result = await db.execute(
        select(Match)
        .where(Match.job_id == job_id)
        .order_by(Match.rank)
    )
    matches = match_result.scalars().all()

    if not matches:
        return MatchResponse(
            job_id=job_id, job_title=job.title,
            total_candidates_scanned=0, top_matches=[],
        )

    # Fetch candidate details
    candidate_ids = [m.candidate_id for m in matches]
    cand_result = await db.execute(
        select(Candidate).where(Candidate.id.in_(candidate_ids))
    )
    candidates = {c.id: c for c in cand_result.scalars().all()}

    top_matches = [
        CandidateMatchResult(
            rank             = m.rank,
            candidate_id     = m.candidate_id,
            name             = candidates[m.candidate_id].name if m.candidate_id in candidates else "Unknown",
            email            = candidates[m.candidate_id].email if m.candidate_id in candidates else None,
            skills           = candidates[m.candidate_id].skills if m.candidate_id in candidates else [],
            experience_years = candidates[m.candidate_id].experience_years if m.candidate_id in candidates else None,
            education        = candidates[m.candidate_id].education if m.candidate_id in candidates else None,
            final_score      = m.final_score,
            llm_summary      = m.llm_summary,
            score_breakdown  = ScoreBreakdown(
                skill_score      = m.skill_score,
                experience_score = m.experience_score,
                education_score  = m.education_score,
                cert_score       = m.cert_score,
                semantic_score   = m.semantic_score,
            ),
        )
        for m in matches
        if m.candidate_id in candidates
    ]

    return MatchResponse(
        job_id=job_id, job_title=job.title,
        total_candidates_scanned=get_faiss_index().total,
        top_matches=top_matches,
    )
