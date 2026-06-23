"""
app/api/resume.py
------------------
FastAPI router for resume endpoints.

POST   /api/v1/upload-resume        Upload single resume
POST   /api/v1/upload-resumes-bulk  Upload multiple resumes at once
DELETE /api/v1/candidates/all       Wipe all candidates and matches
GET    /api/v1/candidates           List all candidates
GET    /api/v1/candidates/{id}      Get single candidate
GET    /api/v1/candidates/skill/{skill} Filter by skill
"""

import uuid
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


# ─────────────────────────────────────────────────────────────────
#  POST /upload-resume  — single upload
# ─────────────────────────────────────────────────────────────────

@router.post(
    "/upload-resume",
    response_model=CandidateResponse,
    status_code=201,
    summary="Upload a single resume PDF",
)
async def upload_resume(
    file: UploadFile = File(...),
    db:   AsyncSession = Depends(get_db),
) -> CandidateResponse:
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    if len(file_bytes) > settings.max_file_size_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Max {settings.MAX_FILE_SIZE_MB}MB.")

    try:
        candidate = await process_resume(file_bytes, file.filename, db)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Resume processing failed: {}", e)
        raise HTTPException(status_code=500, detail="Resume processing failed.")

    return CandidateResponse(
        id               = candidate.id,
        name             = candidate.name,
        email            = candidate.email,
        phone            = candidate.phone,
        skills           = candidate.skills or [],
        experience_years = candidate.experience_years,
        education        = candidate.education,
        certifications   = candidate.certifications or [],
        file_name        = candidate.file_name,
        created_at       = candidate.created_at,
    )


# ─────────────────────────────────────────────────────────────────
#  POST /upload-resumes-bulk  — multiple uploads
# ─────────────────────────────────────────────────────────────────

@router.post(
    "/upload-resumes-bulk",
    status_code=200,
    summary="Upload multiple resume PDFs at once",
)
async def upload_resumes_bulk(
    files: list[UploadFile] = File(...),
    db:    AsyncSession     = Depends(get_db),
) -> dict:
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per upload.")

    results   = []
    succeeded = 0
    failed    = 0

    for file in files:
        if not file.filename.endswith(".pdf"):
            results.append({"file_name": file.filename, "status": "failed", "error": "Not a PDF file"})
            failed += 1
            continue

        file_bytes = await file.read()
        if len(file_bytes) > settings.max_file_size_bytes:
            results.append({"file_name": file.filename, "status": "failed", "error": f"File too large (max {settings.MAX_FILE_SIZE_MB}MB)"})
            failed += 1
            continue

        try:
            candidate = await process_resume(file_bytes, file.filename, db)
            results.append({
                "file_name":        file.filename,
                "status":           "success",
                "candidate_id":     candidate.id,
                "name":             candidate.name,
                "email":            candidate.email,
                "skills":           (candidate.skills or [])[:10],
                "experience_years": candidate.experience_years,
                "education":        candidate.education,
            })
            succeeded += 1
        except Exception as e:
            logger.error("Bulk upload failed for {}: {}", file.filename, e)
            results.append({"file_name": file.filename, "status": "failed", "error": str(e)})
            failed += 1

    logger.info("Bulk upload complete — {} succeeded, {} failed", succeeded, failed)
    return {"total": len(files), "succeeded": succeeded, "failed": failed, "results": results}


# ─────────────────────────────────────────────────────────────────
#  DELETE /candidates/all  — wipe everything and reset FAISS
# ─────────────────────────────────────────────────────────────────

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
        await db.flush()

        # Reset FAISS index in memory using proper reset()
        from app.services.faiss_index import get_faiss_index
        index = get_faiss_index()
        index.reset()

        logger.info("All candidates and matches deleted")
        return {
            "message":   "All candidates, matches deleted and FAISS index reset.",
            "candidates": 0,
            "matches":    0,
        }
    except Exception as e:
        logger.error("Failed to delete all candidates: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────
#  GET /candidates
# ─────────────────────────────────────────────────────────────────

@router.get("/candidates", response_model=dict, summary="List all candidates")
async def list_all_candidates(
    skip:  int = Query(default=0,  ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db:    AsyncSession = Depends(get_db),
) -> dict:
    total, candidates = await list_candidates(db, skip=skip, limit=limit)
    return {
        "total": total, "skip": skip, "limit": limit,
        "candidates": [
            CandidateResponse(
                id=c.id, name=c.name, email=c.email, phone=c.phone,
                skills=c.skills or [], experience_years=c.experience_years,
                education=c.education, certifications=c.certifications or [],
                file_name=c.file_name, created_at=c.created_at,
            ) for c in candidates
        ],
    }


# ─────────────────────────────────────────────────────────────────
#  GET /candidates/{id}
# ─────────────────────────────────────────────────────────────────

@router.get("/candidates/{candidate_id}", response_model=CandidateResponse, summary="Get a single candidate")
async def get_candidate_by_id(candidate_id: str, db: AsyncSession = Depends(get_db)) -> CandidateResponse:
    candidate = await get_candidate(candidate_id, db)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return CandidateResponse(
        id=candidate.id, name=candidate.name, email=candidate.email,
        phone=candidate.phone, skills=candidate.skills or [],
        experience_years=candidate.experience_years, education=candidate.education,
        certifications=candidate.certifications or [], file_name=candidate.file_name,
        created_at=candidate.created_at,
    )


# ─────────────────────────────────────────────────────────────────
#  GET /candidates/skill/{skill}
# ─────────────────────────────────────────────────────────────────

@router.get("/candidates/skill/{skill}", response_model=list[CandidateResponse], summary="Filter by skill")
async def get_by_skill(skill: str, limit: int = Query(default=20, ge=1, le=100), db: AsyncSession = Depends(get_db)) -> list[CandidateResponse]:
    candidates = await get_candidates_by_skill(skill, db, limit=limit)
    return [
        CandidateResponse(
            id=c.id, name=c.name, email=c.email, phone=c.phone,
            skills=c.skills or [], experience_years=c.experience_years,
            education=c.education, certifications=c.certifications or [],
            file_name=c.file_name, created_at=c.created_at,
        ) for c in candidates
    ]