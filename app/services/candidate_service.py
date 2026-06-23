"""
app/services/candidate_service.py
-----------------------------------
Orchestrates the full resume processing pipeline:

  1. Parse PDF → raw text
  2. Extract structured fields (name, email, phone, education)
  3. Extract skills (keyword + spaCy)
  4. Generate embedding (sentence transformer)
  5. Store in PostgreSQL (upsert by filename — allows same person
     to have multiple resumes for different roles)
  6. Add to FAISS index

Deduplication strategy:
  - Same filename uploaded again → update existing record
  - Same email but different filename → create new candidate
  This allows comparing e.g. "John_DataAnalyst.pdf" vs "John_MLEngineer.pdf"
"""

import uuid
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, cast, String

from app.models.candidate import Candidate
from app.schemas.candidate import CandidateCreate, ParsedResume, CandidateResponse
from app.services import parser, skill_extractor, embedder
from app.services.faiss_index import get_faiss_index
from app.core.config import settings


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

    Deduplication: by filename only.
    - Same filename → update existing record
    - Different filename (even same email) → new candidate record
    This allows one person to have multiple role-specific resumes.
    """
    logger.info("Processing resume: {}", filename)

    # ── Step 1: Parse PDF → raw text ─────────────────────────────
    raw_text = parser.extract_text(file_bytes, filename)
    logger.debug("Extracted {} chars from {}", len(raw_text), filename)

    # ── Step 2: Extract structured fields ─────────────────────────
    name           = parser.extract_name(raw_text)
    email          = parser.extract_email(raw_text)
    phone          = parser.extract_phone(raw_text)
    education      = parser.extract_education(raw_text)
    exp_years      = parser.extract_experience_years(raw_text)
    certifications = parser.extract_certifications(raw_text)
    logger.info("EXPERIENCE FOUND = {}", exp_years)

    # ── Step 3: Extract skills ────────────────────────────────────
    skills = skill_extractor.extract_all_skills(raw_text)
    logger.info("Skills found ({}): {}", len(skills), skills[:10])

    # ── Step 4: Generate embedding ────────────────────────────────
    embedding_text = embedder.prepare_resume_text(raw_text, skills)
    embedding      = await embedder.generate_embedding_async(embedding_text)
    logger.debug("Embedding generated — dim={}", len(embedding))

    # ── Step 5: Save to file ──────────────────────────────────────
    file_path = None
    try:
        save_path = settings.upload_path / f"{uuid.uuid4()}_{filename}"
        save_path.write_bytes(file_bytes)
        file_path = str(save_path)
    except Exception as e:
        logger.warning("Could not save PDF file: {}", e)

    # ── Step 6: Upsert by filename ────────────────────────────────
    # Same filename → update; different filename → new record
    result    = await db.execute(
        select(Candidate).where(Candidate.file_name == filename)
    )
    candidate = result.scalar_one_or_none()
    is_update = candidate is not None

    if is_update:
        candidate.name             = name
        candidate.email            = email
        candidate.phone            = phone
        candidate.raw_text         = raw_text
        candidate.file_path        = file_path or candidate.file_path
        candidate.skills           = skills
        candidate.experience_years = exp_years
        candidate.experience_raw   = f"{exp_years} years" if exp_years else None
        candidate.education        = education
        candidate.certifications   = certifications
        candidate.embedding        = embedding
        logger.info(
            "Candidate updated (same filename) — id={} file={}",
            candidate.id, filename,
        )
    else:
        candidate = Candidate(
            id               = str(uuid.uuid4()),
            name             = name,
            email            = email,
            phone            = phone,
            raw_text         = raw_text,
            file_name        = filename,
            file_path        = file_path,
            skills           = skills,
            experience_years = exp_years,
            experience_raw   = f"{exp_years} years" if exp_years else None,
            education        = education,
            certifications   = certifications,
            embedding        = embedding,
        )
        db.add(candidate)

    await db.flush()

    logger.info(
        "Candidate {} — id={} name={} file={} skills={} embedding={}dims",
        "updated" if is_update else "saved",
        candidate.id, name, filename, len(skills), len(embedding),
    )

    # ── Step 7: Add/update in FAISS index ────────────────────────
    try:
        index = get_faiss_index()
        if candidate.embedding:
            index.add(
                candidate_id=candidate.id,
                embedding=candidate.embedding,
            )
            logger.info(
                "Candidate added to FAISS index — id={}",
                candidate.id,
            )
        else:
            logger.warning(
                "Skipping FAISS indexing — no embedding for id={}",
                candidate.id,
            )
    except Exception as e:
        logger.error("FAISS indexing failed for id={}: {}", candidate.id, e)

    return candidate


# ─────────────────────────────────────────────────────────────────
#  Read operations
# ─────────────────────────────────────────────────────────────────

async def get_candidate(candidate_id: str, db: AsyncSession) -> Candidate | None:
    result = await db.execute(
        select(Candidate).where(Candidate.id == candidate_id)
    )
    return result.scalar_one_or_none()


async def list_candidates(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 20,
) -> tuple[int, list[Candidate]]:
    count_result = await db.execute(select(func.count(Candidate.id)))
    total        = count_result.scalar_one()

    result = await db.execute(
        select(Candidate)
        .order_by(Candidate.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return total, list(result.scalars().all())


async def get_candidates_by_skill(
    skill: str,
    db: AsyncSession,
    limit: int = 20,
) -> list[Candidate]:
    result = await db.execute(
        select(Candidate)
        .where(cast(Candidate.skills, String).ilike(f"%{skill}%"))
        .limit(limit)
    )
    return list(result.scalars().all())