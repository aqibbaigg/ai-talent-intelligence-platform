"""
app/services/ats_scorer.py
---------------------------
ATS (Applicant Tracking System) score calculator.

ATS score measures how well a resume is optimized for a specific job.
Unlike the match score (semantic similarity + weighted scoring),
ATS score focuses on keyword and requirement coverage — exactly what
real ATS systems check before a human ever sees the resume.

Formula:
  ATS Score = 40% Keyword match (job description keywords in resume)
            + 30% Required skills coverage
            + 15% Experience requirement met
            + 10% Education requirement met
            +  5% Certifications present

Score range: 0–100
"""

import re
from typing import Optional
from app.models.candidate import Candidate
from app.models.job import Job


def compute_ats_score(candidate: Candidate, job: Job) -> float:
    """
    Compute ATS score for a candidate against a job posting.

    Returns float 0–100.
    """

    # ── 1. Keyword match (40%) ─────────────────────────────────
    # Extract keywords from job description + title
    keyword_score = _keyword_match(
        resume_text = candidate.raw_text or "",
        job_text    = f"{job.title} {job.description} {' '.join(job.required_skills or [])}",
    )

    # ── 2. Required skills coverage (30%) ──────────────────────
    skill_score = _skills_coverage(
        candidate_skills = candidate.skills or [],
        required_skills  = job.required_skills or [],
    )

    # ── 3. Experience requirement met (15%) ────────────────────
    exp_score = _experience_match(
        candidate_years = candidate.experience_years,
        job_min         = job.experience_min,
        job_max         = job.experience_max,
    )

    # ── 4. Education requirement met (10%) ─────────────────────
    edu_score = _education_match(
        candidate_education = candidate.education,
        job_education       = job.education,
    )

    # ── 5. Certifications (5%) ─────────────────────────────────
    cert_score = _cert_match(
        candidate_certs = candidate.certifications or [],
        required_skills = job.required_skills or [],
    )

    # Weighted final ATS score
    ats = (
        keyword_score * 0.40
        + skill_score * 0.30
        + exp_score   * 0.15
        + edu_score   * 0.10
        + cert_score  * 0.05
    )

    return round(ats, 1)


# ─────────────────────────────────────────────────────────────────
#  Sub-scorers
# ─────────────────────────────────────────────────────────────────

def _keyword_match(resume_text: str, job_text: str) -> float:
    """
    Check what percentage of significant job keywords appear in the resume.
    Ignores common stop words — focuses on technical and role-specific terms.
    """
    stop_words = {
        "the","and","or","in","on","at","to","for","of","a","an","is","are",
        "was","were","be","been","being","have","has","had","do","does","did",
        "will","would","could","should","may","might","with","from","by","this",
        "that","we","you","they","their","our","your","its","as","if","but",
        "not","no","so","than","then","when","where","who","what","how","all",
        "both","each","more","most","other","some","such","only","own","same",
        "also","must","just","about","into","through","during","before","after",
    }

    # Extract words from job text (3+ chars, not stop words)
    job_words = set(
        w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', job_text)
        if w.lower() not in stop_words
    )

    if not job_words:
        return 80.0

    resume_lower = resume_text.lower()
    matched = sum(1 for w in job_words if w in resume_lower)

    return round(min(100.0, (matched / len(job_words)) * 100), 1)


def _skills_coverage(
    candidate_skills: list[str],
    required_skills:  list[str],
) -> float:
    """Percentage of required skills found in candidate skills (fuzzy match)."""
    if not required_skills:
        return 100.0

    candidate_lower = {s.lower() for s in candidate_skills}

    matched = 0
    for req in required_skills:
        req_l = req.lower()
        # Exact or substring match
        if any(req_l in c or c in req_l for c in candidate_lower):
            matched += 1

    return round((matched / len(required_skills)) * 100, 1)


def _experience_match(
    candidate_years: Optional[float],
    job_min:         Optional[int],
    job_max:         Optional[int],
) -> float:
    if candidate_years is None:
        return 60.0  # unknown — neutral

    if job_min is None and job_max is None:
        return 90.0  # no requirement

    min_req = job_min or 0
    max_req = job_max or 99

    if min_req <= candidate_years <= max_req + 2:
        return 100.0
    elif candidate_years < min_req:
        ratio = candidate_years / max(min_req, 1)
        return round(min(85.0, ratio * 100), 1)
    else:
        return 85.0  # overqualified


def _education_match(
    candidate_education: Optional[str],
    job_education:       Optional[str],
) -> float:
    if not job_education:
        return 90.0
    if not candidate_education:
        return 50.0

    tiers = [
        ["diploma", "certificate", "intermediate"],
        ["bachelor", "b.tech", "btech", "b.e", "b.sc", "bca", "engineering", "computer science"],
        ["master", "m.tech", "mtech", "m.s", "m.sc", "mca", "mba"],
        ["phd", "ph.d", "doctorate"],
    ]

    def get_tier(text: str) -> int:
        t = text.lower()
        for i, tier in enumerate(tiers):
            if any(k in t for k in tier):
                return i
        return 1

    c_tier = get_tier(candidate_education)
    j_tier = get_tier(job_education)

    if c_tier >= j_tier:
        return 100.0
    elif c_tier == j_tier - 1:
        return 70.0
    else:
        return 40.0


def _cert_match(
    candidate_certs: list[str],
    required_skills: list[str],
) -> float:
    if not candidate_certs:
        return 70.0  # no certs — mild penalty
    if not required_skills:
        return 90.0

    req_lower = " ".join(required_skills).lower()
    relevant  = sum(
        1 for cert in candidate_certs
        if any(w in req_lower for w in cert.lower().split())
    )
    return min(100.0, 70.0 + relevant * 15)
