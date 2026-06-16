"""
app/api/resume.py
------------------
FastAPI router for all resume-related endpoints.

Endpoints
---------
POST /upload-resume     Upload and process a PDF resume
GET  /candidates        List all candidates (paginated)
GET  /candidates/{id}   Get a single candidate
GET  /candidates/skill/{skill}  Filter by skill
"""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Query
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.config import settings
from app.core.database import get_db
from app.schemas.candidate import CandidateResponse, CandidateList, ErrorResponse
from app.services import candidate_service

router = APIRouter(prefix="/api/v1", tags=["resumes"])


# ─────────────────────────────────────────────────────────────────
#  POST /upload-resume  ← THE CORE WEEK 1 ENDPOINT
# ─────────────────────────────────────────────────────────────────

@router.post(
    "/upload-resume",
    response_model=CandidateResponse,
    status_code=201,
    summary="Upload and process a resume PDF",
    description="""
    Upload a candidate's resume (PDF only).

    This endpoint:
    1. Extracts text from the PDF (pdfplumber → PyPDF2 fallback)
    2. Parses structured fields (name, email, phone, education)
    3. Extracts skills (keyword matching + spaCy NER)
    4. Generates a 384-dimensional sentence embedding
    5. Stores everything in PostgreSQL

    Returns the parsed candidate profile.
    """,
)
async def upload_resume(
    file: UploadFile = File(..., description="Resume PDF file"),
    db: AsyncSession = Depends(get_db),
) -> CandidateResponse:
    """
    Full resume processing pipeline in one endpoint.
    """

    # ── Validate file type ────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file.",
        )

    # ── Validate file size ────────────────────────────────────────
    file_bytes = await file.read()
    if len(file_bytes) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB.",
        )
    if len(file_bytes) < 100:
        raise HTTPException(
            status_code=400,
            detail="File appears to be empty or corrupt.",
        )

    logger.info(
        "Resume upload received — file={} size={}KB",
        file.filename, len(file_bytes) // 1024,
    )

    # ── Process resume (parse → extract → embed → store) ──────────
    try:
        candidate = await candidate_service.process_resume(
            file_bytes=file_bytes,
            filename=file.filename,
            db=db,
        )
    except ValueError as e:
        # Parser couldn't extract text (scanned PDF, etc.)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Resume processing failed: {}", e)
        raise HTTPException(
            status_code=500,
            detail="Resume processing failed. Please try again.",
        )

    # ── Build response ─────────────────────────────────────────────
    return CandidateResponse(
        id                  = candidate.id,
        name                = candidate.name,
        email               = candidate.email,
        phone               = candidate.phone,
        file_name           = candidate.file_name,
        skills              = candidate.skills,
        experience_years    = candidate.experience_years,
        experience_raw      = candidate.experience_raw,
        education           = candidate.education,
        certifications      = candidate.certifications,
        embedding_generated = candidate.embedding is not None,
        created_at          = candidate.created_at,
    )


# ─────────────────────────────────────────────────────────────────
#  GET /candidates  — paginated list
# ─────────────────────────────────────────────────────────────────

@router.get(
    "/candidates",
    response_model=CandidateList,
    summary="List all candidates",
)
async def list_candidates(
    skip:  int = Query(default=0, ge=0, description="Pagination offset"),
    limit: int = Query(default=20, ge=1, le=100, description="Results per page"),
    db: AsyncSession = Depends(get_db),
) -> CandidateList:
    total, candidates = await candidate_service.list_candidates(db, skip, limit)
    return CandidateList(
        total=total,
        candidates=[
            CandidateResponse(
                id                  = c.id,
                name                = c.name,
                email               = c.email,
                phone               = c.phone,
                file_name           = c.file_name,
                skills              = c.skills,
                experience_years    = c.experience_years,
                experience_raw      = c.experience_raw,
                education           = c.education,
                certifications      = c.certifications,
                embedding_generated = c.embedding is not None,
                created_at          = c.created_at,
            )
            for c in candidates
        ],
    )


# ─────────────────────────────────────────────────────────────────
#  GET /candidates/{id}  — single candidate
# ─────────────────────────────────────────────────────────────────

@router.get(
    "/candidates/{candidate_id}",
    response_model=CandidateResponse,
    summary="Get candidate by ID",
)
async def get_candidate(
    candidate_id: str,
    db: AsyncSession = Depends(get_db),
) -> CandidateResponse:
    candidate = await candidate_service.get_candidate(candidate_id, db)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    return CandidateResponse(
        id                  = candidate.id,
        name                = candidate.name,
        email               = candidate.email,
        phone               = candidate.phone,
        file_name           = candidate.file_name,
        skills              = candidate.skills,
        experience_years    = candidate.experience_years,
        experience_raw      = candidate.experience_raw,
        education           = candidate.education,
        certifications      = candidate.certifications,
        embedding_generated = candidate.embedding is not None,
        created_at          = candidate.created_at,
    )


# ─────────────────────────────────────────────────────────────────
#  GET /candidates/skill/{skill}  — filter by skill
# ─────────────────────────────────────────────────────────────────

@router.get(
    "/candidates/skill/{skill}",
    response_model=list[CandidateResponse],
    summary="Find candidates with a specific skill",
)
async def get_by_skill(
    skill: str,
    db: AsyncSession = Depends(get_db),
) -> list[CandidateResponse]:
    candidates = await candidate_service.get_candidates_by_skill(skill, db)
    return [
        CandidateResponse(
            id                  = c.id,
            name                = c.name,
            email               = c.email,
            phone               = c.phone,
            file_name           = c.file_name,
            skills              = c.skills,
            experience_years    = c.experience_years,
            experience_raw      = c.experience_raw,
            education           = c.education,
            certifications      = c.certifications,
            embedding_generated = c.embedding is not None,
            created_at          = c.created_at,
        )
        for c in candidates
    ]
