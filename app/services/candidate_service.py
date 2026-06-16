"""
app/services/candidate_service.py
-----------------------------------
Orchestrates the full resume processing pipeline:

  1. Parse PDF → raw text
  2. Extract structured fields (name, email, phone, education)
  3. Extract skills (keyword + spaCy)
  4. Generate embedding (sentence transformer)
  5. Store in PostgreSQL

This is the service layer — it has no knowledge of HTTP (FastAPI).
The route handler calls this service and handles HTTP responses.
"""

import uuid
from pathlib import Path

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.candidate import Candidate
from app.schemas.candidate import CandidateCreate, ParsedResume, CandidateResponse
from app.services import parser, skill_extractor, embedder
from app.core.config import settings
from sqlalchemy import select, func, cast, String

# ─────────────────────────────────────────────────────────────────
#  Process resume — main pipeline
# ─────────────────────────────────────────────────────────────────

async def process_resume(
    file_bytes: bytes,
    filename: str,
    db: AsyncSession,
) -> Candidate:
    """
    Full resume processing pipeline.

    Parameters
    ----------
    file_bytes : raw PDF bytes from upload
    filename   : original filename (for logging + storage)
    db         : async DB session

    Returns
    -------
    Candidate  — the saved database record

    Raises
    ------
    ValueError  if PDF cannot be parsed
    Exception   if DB insert fails
    """
    logger.info("Processing resume: {}", filename)

    # ── Step 1: Parse PDF → raw text ─────────────────────────────
    raw_text = parser.extract_text(file_bytes, filename)
    logger.debug("Extracted {} chars from {}", len(raw_text), filename)

    # ── Step 2: Extract structured fields ─────────────────────────
    name          = parser.extract_name(raw_text)
    email         = parser.extract_email(raw_text)
    phone         = parser.extract_phone(raw_text)
    education     = parser.extract_education(raw_text)
    exp_years     = parser.extract_experience_years(raw_text)
    logger.info("EXPERIENCE FOUND = {}", exp_years)
    certifications = parser.extract_certifications(raw_text)

    logger.debug(
        "Extracted fields — name={} email={} exp={}yrs skills=?",
        name, email, exp_years,
    )

    # ── Step 3: Extract skills ────────────────────────────────────
    skills = skill_extractor.extract_all_skills(raw_text)
    logger.info("Skills found ({}): {}", len(skills), skills[:10])

    # ── Step 4: Generate embedding ────────────────────────────────
    embedding_text = embedder.prepare_resume_text(raw_text, skills)
    embedding      = await embedder.generate_embedding_async(embedding_text)
    logger.debug("Embedding generated — dim={}", len(embedding))

    # ── Step 5: Save to file (optional, for audit trail) ──────────
    file_path = None
    try:
        save_path = settings.upload_path / f"{uuid.uuid4()}_{filename}"
        save_path.write_bytes(file_bytes)
        file_path = str(save_path)
    except Exception as e:
        logger.warning("Could not save PDF file: {}", e)

    # ── Step 6: Store in PostgreSQL ───────────────────────────────
    candidate = Candidate(
        id              = str(uuid.uuid4()),
        name            = name,
        email           = email,
        phone           = phone,
        raw_text        = raw_text,
        file_name       = filename,
        file_path       = file_path,
        skills          = skills,
        experience_years = exp_years,
        experience_raw  = f"{exp_years} years" if exp_years else None,
        education       = education,
        certifications  = certifications,
        embedding       = embedding,
    )

    db.add(candidate)
    await db.flush()    # write to DB without committing (commit happens in get_db)

    logger.info(
        "Candidate saved — id={} name={} skills={} embedding={}dims",
        candidate.id, name, len(skills), len(embedding),
    )
    return candidate


# ─────────────────────────────────────────────────────────────────
#  Read operations
# ─────────────────────────────────────────────────────────────────

async def get_candidate(candidate_id: str, db: AsyncSession) -> Candidate | None:
    """Fetch a single candidate by ID."""
    result = await db.execute(
        select(Candidate).where(Candidate.id == candidate_id)
    )
    return result.scalar_one_or_none()


async def list_candidates(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 20,
) -> tuple[int, list[Candidate]]:
    """
    Paginated list of all candidates.

    Returns
    -------
    (total_count, candidates_on_this_page)
    """
    # Total count
    count_result = await db.execute(
        select(func.count(Candidate.id))
    )
    total = count_result.scalar_one()

    # Paginated results
    result = await db.execute(
        select(Candidate)
        .order_by(Candidate.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    candidates = result.scalars().all()

    return total, list(candidates)


async def get_candidates_by_skill(
    skill: str,
    db: AsyncSession,
    limit: int = 20,
) -> list[Candidate]:

    result = await db.execute(
        select(Candidate)
        .where(
            cast(Candidate.skills, String).ilike(f"%{skill}%")
        )
        .limit(limit)
    )

    return list(result.scalars().all())
