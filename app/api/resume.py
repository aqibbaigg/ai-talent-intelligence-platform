"""
app/api/resume.py
------------------
FastAPI router for resume endpoints.
"""

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from loguru import logger

from app.core.database import get_db
from app.core.config import settings
from app.schemas.candidate import CandidateResponse
from app.services.candidate_service import (
    process_resume,
    get_candidate,
    list_candidates,
    get_candidates_by_skill,
)

router = APIRouter(prefix="/api/v1", tags=["resumes & candidates"])


def build_candidate_response(candidate) -> CandidateResponse:
    return CandidateResponse(
        id=candidate.id,
        name=candidate.name,
        email=candidate.email,
        phone=candidate.phone,
        skills=candidate.skills or [],
        experience_years=candidate.experience_years or 0,
        experience_raw=candidate.experience_raw,
        education=candidate.education,
        certifications=candidate.certifications or [],
        file_name=candidate.file_name,
        embedding_generated=bool(candidate.embedding),
        created_at=candidate.created_at,
    )


@router.options("/upload-resumes-bulk")
async def upload_resumes_bulk_options():
    return {}


@router.post(
    "/upload-resume",
    response_model=CandidateResponse,
    status_code=201,
    summary="Upload a single resume PDF",
)
async def upload_resume(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> CandidateResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()

    if len(file_bytes) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {settings.MAX_FILE_SIZE_MB}MB.",
        )

    try:
        candidate = await process_resume(file_bytes, file.filename, db)
        return build_candidate_response(candidate)

    except ValueError as e:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        await db.rollback()
        logger.exception("Resume processing failed: {}", e)
        raise HTTPException(status_code=500, detail="Resume processing failed.")


@router.post(
    "/upload-resumes-bulk",
    status_code=200,
    summary="Upload multiple resume PDFs at once",
)
async def upload_resumes_bulk(
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per upload.")

    results = []
    succeeded = 0
    failed = 0

    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            results.append(
                {
                    "file_name": file.filename,
                    "status": "failed",
                    "error": "Not a PDF file",
                }
            )
            failed += 1
            continue

        file_bytes = await file.read()

        if len(file_bytes) > settings.max_file_size_bytes:
            results.append(
                {
                    "file_name": file.filename,
                    "status": "failed",
                    "error": f"File too large (max {settings.MAX_FILE_SIZE_MB}MB)",
                }
            )
            failed += 1
            continue

        try:
            candidate = await process_resume(file_bytes, file.filename, db)
            results.append(
                {
                    "file_name": file.filename,
                    "status": "success",
                    "candidate_id": candidate.id,
                    "name": candidate.name,
                    "email": candidate.email,
                    "skills": candidate.skills or [],
                    "experience_years": candidate.experience_years or 0,
                    "experience_raw": candidate.experience_raw,
                    "education": candidate.education,
                    "embedding_generated": bool(candidate.embedding),
                }
            )
            succeeded += 1

        except Exception as e:
            await db.rollback()
            logger.exception("Bulk upload failed for {}: {}", file.filename, e)
            results.append(
                {
                    "file_name": file.filename,
                    "status": "failed",
                    "error": str(e),
                }
            )
            failed += 1

    logger.info("Bulk upload complete — {} succeeded, {} failed", succeeded, failed)

    return {
        "total": len(files),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }


@router.delete(
    "/candidates/all",
    status_code=200,
    summary="Delete all candidates and matches, reset FAISS index",
)
async def delete_all_candidates(
    db: AsyncSession = Depends(get_db),
) -> dict:
    try:
        await db.execute(text("DELETE FROM matches"))
        await db.execute(text("DELETE FROM candidates"))
        await db.commit()

        from app.services.faiss_index import get_faiss_index

        index = get_faiss_index()
        index.reset()

        logger.info("All candidates and matches deleted")

        return {
            "message": "All candidates, matches deleted and FAISS index reset.",
            "candidates": 0,
            "matches": 0,
        }

    except Exception as e:
        await db.rollback()
        logger.exception("Failed to delete all candidates: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/candidates", response_model=dict, summary="List all candidates")
async def list_all_candidates(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> dict:
    total, candidates = await list_candidates(db, skip=skip, limit=limit)

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "candidates": [build_candidate_response(c) for c in candidates],
    }


@router.get(
    "/candidates/{candidate_id}",
    response_model=CandidateResponse,
    summary="Get a single candidate",
)
async def get_candidate_by_id(
    candidate_id: str,
    db: AsyncSession = Depends(get_db),
) -> CandidateResponse:
    candidate = await get_candidate(candidate_id, db)

    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    return build_candidate_response(candidate)


@router.get(
    "/candidates/skill/{skill}",
    response_model=list[CandidateResponse],
    summary="Filter by skill",
)
async def get_by_skill(
    skill: str,
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> list[CandidateResponse]:
    candidates = await get_candidates_by_skill(skill, db, limit=limit)
    return [build_candidate_response(c) for c in candidates]