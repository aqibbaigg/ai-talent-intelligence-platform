"""
app/services/job_service.py
-----------------------------
Orchestrates:
  - Job creation (embed description → store)
  - Candidate matching (FAISS search → score → rank → save)
"""

import uuid
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.job import Job
from app.models.match import Match
from app.models.candidate import Candidate
from app.schemas.job import JobCreate, MatchResponse, CandidateMatchResult, ScoreBreakdown
from app.services import embedder
from app.services.faiss_index import get_faiss_index
from app.services.scorer import compute_score
from app.services.ats_scorer import compute_ats_score


# ─────────────────────────────────────────────────────────────────
#  Create job
# ─────────────────────────────────────────────────────────────────

async def create_job(data: JobCreate, db: AsyncSession) -> Job:
    embed_text = (
        f"Job Title: {data.title}\n\n"
        f"Required Skills: {', '.join(data.required_skills)}\n\n"
        f"Description: {data.description}"
    )
    embedding = await embedder.generate_embedding_async(embed_text)

    job = Job(
        id                  = str(uuid.uuid4()),
        title               = data.title,
        company             = data.company,
        description         = data.description,
        required_skills     = data.required_skills,
        nice_to_have_skills = data.nice_to_have_skills,
        experience_min      = data.experience_min,
        experience_max      = data.experience_max,
        education           = data.education,
        location            = data.location,
        job_type            = data.job_type,
        embedding           = embedding,
    )

    db.add(job)
    await db.flush()
    logger.info("Job created — id={} title={}", job.id, job.title)
    return job


# ─────────────────────────────────────────────────────────────────
#  Match candidates to a job
# ─────────────────────────────────────────────────────────────────

async def match_candidates(
    job_id: str,
    db:     AsyncSession,
    top_k:  int = 10,
) -> MatchResponse:

    # ── Fetch job ─────────────────────────────────────────────────
    result = await db.execute(select(Job).where(Job.id == job_id))
    job    = result.scalar_one_or_none()
    if not job:
        raise ValueError(f"Job not found: {job_id}")
    if not job.embedding:
        raise ValueError("Job has no embedding — re-create the job to generate one")

    # ── FAISS search ──────────────────────────────────────────────
    index = get_faiss_index()
    if not index.is_ready or index.total == 0:
        raise RuntimeError("FAISS index is empty. Upload some resumes first.")

    raw_results = index.search(job.embedding, top_k=min(top_k * 3, index.total))
    logger.info(
        "FAISS search for job={} — found {} candidates before scoring",
        job_id, len(raw_results),
    )

    if not raw_results:
        return MatchResponse(
            job_id=job_id, job_title=job.title,
            total_candidates_scanned=index.total, top_matches=[],
        )

    # ── Fetch candidates from DB ──────────────────────────────────
    candidate_ids      = [cid for cid, _ in raw_results]
    semantic_score_map = {cid: score for cid, score in raw_results}

    db_result  = await db.execute(
        select(Candidate).where(Candidate.id.in_(candidate_ids))
    )
    candidates = {c.id: c for c in db_result.scalars().all()}

    # ── Score and rank ────────────────────────────────────────────
    scored: list[tuple[Candidate, object, float]] = []

    for cid in candidate_ids:
        candidate = candidates.get(cid)
        if not candidate:
            continue
        sem_score = semantic_score_map.get(cid, 0.0)
        breakdown = compute_score(candidate, job, sem_score)
        ats_score = compute_ats_score(candidate, job)
        scored.append((candidate, breakdown, ats_score))

    # Sort by final score descending
    scored.sort(key=lambda x: x[1].final_score, reverse=True)
    scored = scored[:top_k]

    # ── Save matches to DB ────────────────────────────────────────
    from sqlalchemy import delete
    await db.execute(delete(Match).where(Match.job_id == job_id))

    match_results: list[CandidateMatchResult] = []
    for rank, (candidate, breakdown, ats_score) in enumerate(scored, start=1):
        match = Match(
            id               = str(uuid.uuid4()),
            candidate_id     = candidate.id,
            job_id           = job_id,
            final_score      = breakdown.final_score,
            rank             = rank,
            skill_score      = breakdown.skill_score,
            experience_score = breakdown.experience_score,
            education_score  = breakdown.education_score,
            cert_score       = breakdown.cert_score,
            semantic_score   = breakdown.semantic_score,
        )
        db.add(match)

        match_results.append(CandidateMatchResult(
            rank             = rank,
            candidate_id     = candidate.id,
            name             = candidate.name,
            email            = candidate.email,
            skills           = candidate.skills or [],
            experience_years = candidate.experience_years,
            education        = candidate.education,
            final_score      = breakdown.final_score,
            ats_score        = ats_score,
            score_breakdown  = ScoreBreakdown(
                skill_score      = breakdown.skill_score,
                experience_score = breakdown.experience_score,
                education_score  = breakdown.education_score,
                cert_score       = breakdown.cert_score,
                semantic_score   = breakdown.semantic_score,
            ),
        ))

    await db.flush()
    logger.info(
        "Matching complete — job={} top candidate={} score={} ats={}",
        job_id,
        match_results[0].name if match_results else "none",
        match_results[0].final_score if match_results else 0,
        match_results[0].ats_score if match_results else 0,
    )

    return MatchResponse(
        job_id                   = job_id,
        job_title                = job.title,
        total_candidates_scanned = index.total,
        top_matches              = match_results,
    )