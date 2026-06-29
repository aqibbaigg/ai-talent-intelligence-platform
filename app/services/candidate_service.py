"""
app/services/candidate_service.py
-----------------------------------
Resume processing pipeline.

Railway-safe version:
- Uses keyword-based skill extraction from skill_extractor.py
- Keeps embedding as dummy vector for now
"""

import uuid
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, cast, String

from app.models.candidate import Candidate
from app.services import parser, skill_extractor,embedder 
from app.services.faiss_index import get_faiss_index
from app.core.config import settings


async def process_resume(
    file_bytes: bytes,
    filename: str,
    db: AsyncSession,
) -> Candidate:
    try:
        logger.info("Processing resume: {}", filename)

        raw_text = parser.extract_text(file_bytes, filename)
        logger.debug("Extracted {} chars from {}", len(raw_text), filename)

        if not raw_text or len(raw_text.strip()) < 20:
            raise ValueError("Resume text extraction failed or text is too short")

        name = parser.extract_name(raw_text)
        email = parser.extract_email(raw_text)
        phone = parser.extract_phone(raw_text)
        education = parser.extract_education(raw_text)
        exp_years = parser.extract_experience_years(raw_text)
        certifications = parser.extract_certifications(raw_text)

        logger.info("EXPERIENCE FOUND = {}", exp_years)

        # Step 3: Railway-safe skill extraction
        logger.info("Starting skill extraction...")
        skills = skill_extractor.extract_all_skills(raw_text)
        logger.info("Skills found ({}): {}", len(skills), skills[:10])

        # Step 4: Generate semantic embedding
        logger.info("Starting embedding generation...")

        embedding_text = embedder.prepare_resume_text(
        raw_text=raw_text,
        skills=skills,
)

        embedding = await embedder.generate_embedding_async(embedding_text)

        logger.info(
        "Embedding generated successfully — dim={}",
        len(embedding),
)

        file_path = None
        try:
            settings.upload_path.mkdir(parents=True, exist_ok=True)
            safe_filename = filename.replace("/", "_").replace("\\", "_")
            save_path = settings.upload_path / f"{uuid.uuid4()}_{safe_filename}"
            save_path.write_bytes(file_bytes)
            file_path = str(save_path)
            logger.info("Resume file saved at {}", file_path)
        except Exception as e:
            logger.warning("Could not save PDF file: {}", e)

        result = await db.execute(
            select(Candidate).where(Candidate.file_name == filename)
        )
        candidate = result.scalar_one_or_none()
        is_update = candidate is not None

        if is_update:
            candidate.name = name
            candidate.email = email
            candidate.phone = phone
            candidate.raw_text = raw_text
            candidate.file_path = file_path or candidate.file_path
            candidate.skills = skills
            candidate.experience_years = exp_years
            candidate.experience_raw = f"{exp_years} years" if exp_years else None
            candidate.education = education
            candidate.certifications = certifications
            candidate.embedding = embedding
            logger.info("Candidate updated — id={} file={}", candidate.id, filename)
        else:
            candidate = Candidate(
                id=str(uuid.uuid4()),
                name=name,
                email=email,
                phone=phone,
                raw_text=raw_text,
                file_name=filename,
                file_path=file_path,
                skills=skills,
                experience_years=exp_years,
                experience_raw=f"{exp_years} years" if exp_years else None,
                education=education,
                certifications=certifications,
                embedding=embedding,
            )
            db.add(candidate)

        await db.flush()
        await db.commit()
        await db.refresh(candidate)

        logger.info(
            "Candidate {} successfully — id={} name={} file={} skills={} embedding={}dims",
            "updated" if is_update else "saved",
            candidate.id,
            candidate.name,
            filename,
            len(skills),
            len(embedding),
        )

        try:
            index = get_faiss_index()
            if candidate.embedding:
                index.add(
                    candidate_id=candidate.id,
                    embedding=candidate.embedding,
                )
                logger.info("Candidate added to FAISS index — id={}", candidate.id)
        except Exception as e:
            logger.error("FAISS indexing failed for id={}: {}", candidate.id, e)

        return candidate

    except Exception as e:
        logger.exception("Resume processing failed for file {}: {}", filename, e)
        await db.rollback()
        raise


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
    total = count_result.scalar_one()

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