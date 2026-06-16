"""
tests/test_resume_upload.py
-----------------------------
Tests for POST /api/v1/upload-resume.

Run with:
    pytest tests/ -v

Uses httpx.AsyncClient with FastAPI's ASGI transport — no real
server needed, no real DB needed (uses SQLite in-memory for tests).
"""

import io
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from main import app


# ── Test PDF fixtures ──────────────────────────────────────────────

def make_fake_pdf_bytes() -> bytes:
    """
    Minimal valid PDF with resume-like text.
    pdfplumber can parse this without a real PDF library.
    """
    content = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 200 >>
stream
BT /F1 12 Tf 50 750 Td
(John Doe) Tj
0 -20 Td (john.doe@gmail.com) Tj
0 -20 Td (+91 9876543210) Tj
0 -20 Td (Skills: Python, FastAPI, PostgreSQL, Docker, AWS) Tj
0 -20 Td (Experience: 3 years at TechCorp) Tj
0 -20 Td (Education: B.Tech Computer Science) Tj
ET
endstream
endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000516 00000 n
trailer << /Size 6 /Root 1 0 R >>
startxref
600
%%EOF"""
    return content


# ── Tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check():
    """Confirm the app starts and health endpoint works."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_upload_non_pdf_rejected():
    """Non-PDF files must be rejected with 400."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/upload-resume",
            files={"file": ("resume.docx", b"fake content", "application/msword")},
        )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_empty_file_rejected():
    """Empty files must be rejected."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/upload-resume",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_upload_no_file_returns_422():
    """Request with no file must return 422 Unprocessable Entity."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/v1/upload-resume")
    assert response.status_code == 422


# ── Parser unit tests ─────────────────────────────────────────────

def test_extract_email():
    from app.services.parser import extract_email
    text = "Contact me at john.doe@gmail.com for more info"
    assert extract_email(text) == "john.doe@gmail.com"


def test_extract_email_none():
    from app.services.parser import extract_email
    assert extract_email("No email here") is None


def test_extract_phone():
    from app.services.parser import extract_phone
    text = "Call me at +91 9876543210"
    result = extract_phone(text)
    assert result is not None
    assert "9876543210" in result.replace(" ", "")


def test_extract_name():
    from app.services.parser import extract_name
    text = "John Doe\njohn@gmail.com\nSoftware Engineer"
    assert extract_name(text) == "John Doe"


def test_extract_experience_years():
    from app.services.parser import extract_experience_years
    assert extract_experience_years("3 years of experience in Python") == 3.0
    assert extract_experience_years("five years experience") == 5.0
    assert extract_experience_years("No experience mentioned") is None


def test_extract_certifications():
    from app.services.parser import extract_certifications
    text = "I hold AWS Certified Solutions Architect Associate and PMP certifications"
    certs = extract_certifications(text)
    assert len(certs) > 0


# ── Skill extractor unit tests ────────────────────────────────────

def test_keyword_extraction():
    from app.services.skill_extractor import extract_skills_keyword
    text = "Experienced in Python, FastAPI, PostgreSQL, Docker and AWS"
    skills = extract_skills_keyword(text)
    assert "Python" in skills
    assert "FastAPI" in skills
    assert "PostgreSQL" in skills
    assert "Docker" in skills
    assert "AWS" in skills


def test_keyword_no_partial_match():
    """'Java' should not match inside 'JavaScript'."""
    from app.services.skill_extractor import extract_skills_keyword
    text = "Expert in JavaScript development"
    skills = extract_skills_keyword(text)
    assert "JavaScript" in skills
    # Java should NOT be in skills (it's inside JavaScript)
    # This depends on word boundary implementation
    assert skills.count("Java") == 0 or "Java" not in skills


def test_no_skills_empty_text():
    from app.services.skill_extractor import extract_skills_keyword
    assert extract_skills_keyword("") == []


# ── Embedder unit tests ───────────────────────────────────────────

def test_embedding_dimension():
    from app.services.embedder import generate_embedding
    embedding = generate_embedding("Python developer with 3 years experience")
    assert len(embedding) == 384
    assert all(isinstance(v, float) for v in embedding)


def test_embedding_normalised():
    """L2-normalised embedding should have magnitude ~1.0."""
    import math
    from app.services.embedder import generate_embedding
    embedding = generate_embedding("Data scientist with ML expertise")
    magnitude = math.sqrt(sum(v**2 for v in embedding))
    assert abs(magnitude - 1.0) < 0.01


def test_prepare_resume_text():
    from app.services.embedder import prepare_resume_text
    text = "John Doe, Software Engineer"
    skills = ["Python", "FastAPI"]
    result = prepare_resume_text(text, skills)
    assert "Python" in result
    assert "FastAPI" in result
    assert "John Doe" in result
