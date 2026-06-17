"""
app/services/scorer.py
-----------------------
Weighted candidate scoring engine.

Formula (from project spec):
    Final Score = 50% Skill Match
                + 20% Experience
                + 20% Education
                + 10% Certifications

Each sub-score is 0–100. Final score is 0–100.
"""

from dataclasses import dataclass
from typing import Optional

from app.models.candidate import Candidate
from app.models.job import Job


# ─────────────────────────────────────────────────────────────────
#  Score weights (must sum to 1.0)
# ─────────────────────────────────────────────────────────────────

WEIGHTS = {
    "skill":      0.50,
    "experience": 0.20,
    "education":  0.20,
    "cert":       0.10,
}


@dataclass
class ScoreBreakdown:
    skill_score:      float   # 0–100
    experience_score: float   # 0–100
    education_score:  float   # 0–100
    cert_score:       float   # 0–100
    semantic_score:   float   # 0–100 (raw FAISS similarity)
    final_score:      float   # 0–100 (weighted)


# ─────────────────────────────────────────────────────────────────
#  Sub-scorers
# ─────────────────────────────────────────────────────────────────

def _skill_matches(candidate_skill: str, job_skill: str) -> bool:
    """
    Fuzzy skill matching — handles partial matches and common aliases.

    Examples that should match:
      "REST"       ↔ "REST API"
      "Postgres"   ↔ "PostgreSQL"
      "JS"         ↔ "JavaScript"
      "ML"         ↔ "Machine Learning"
    """
    c = candidate_skill.lower().strip()
    j = job_skill.lower().strip()

    # Exact match
    if c == j:
        return True

    # One contains the other (handles "REST" ↔ "REST API")
    if c in j or j in c:
        return True

    # Token overlap — if ALL tokens of the shorter skill appear in the longer
    c_tokens = set(c.split())
    j_tokens = set(j.split())
    shorter = c_tokens if len(c_tokens) <= len(j_tokens) else j_tokens
    longer  = c_tokens if len(c_tokens) >  len(j_tokens) else j_tokens
    if shorter and shorter.issubset(longer):
        return True

    # Common aliases
    aliases = {
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "dl": "deep learning",
        "nlp": "natural language processing",
        "cv": "computer vision",
        "postgres": "postgresql",
        "mongo": "mongodb",
        "k8s": "kubernetes",
        "aws": "amazon web services",
        "gcp": "google cloud",
    }
    c_resolved = aliases.get(c, c)
    j_resolved = aliases.get(j, j)
    if c_resolved == j_resolved:
        return True
    if c_resolved in j_resolved or j_resolved in c_resolved:
        return True

    return False


def score_skills(
    candidate_skills: list[str],
    required_skills:  list[str],
    nice_to_have:     list[str] | None = None,
) -> float:
    """
    Skill match score (0–100).

    Logic:
      - Required skills: worth 80% of skill score
      - Nice-to-have:    worth 20% of skill score
      - Uses fuzzy matching to handle partial names and aliases
    """
    if not required_skills:
        return 100.0   # no requirements = any candidate passes

    # Required skills match (fuzzy)
    required_matched = sum(
        1 for req in required_skills
        if any(_skill_matches(cand, req) for cand in candidate_skills)
    )
    required_score = (required_matched / len(required_skills)) * 100

    # Nice-to-have boost (fuzzy)
    nice_score = 0.0
    if nice_to_have:
        nice_matched = sum(
            1 for req in nice_to_have
            if any(_skill_matches(cand, req) for cand in candidate_skills)
        )
        nice_score = (nice_matched / len(nice_to_have)) * 100

    # Combined: 80% required + 20% nice-to-have
    return round(required_score * 0.8 + nice_score * 0.2, 2)


def score_experience(
    candidate_years: Optional[float],
    job_min:         Optional[int],
    job_max:         Optional[int],
) -> float:
    """
    Experience score (0–100).

    Scoring bands:
      Below minimum:     partial credit (0–80)
      Within range:      100
      Above maximum:     slight penalty (overqualified) → 85–90
      Unknown years:     neutral 50
    """
    if candidate_years is None:
        return 50.0   # unknown — give neutral score

    if job_min is None and job_max is None:
        return 80.0   # no requirement specified

    min_req = job_min or 0
    max_req = job_max or 99

    if candidate_years < min_req:
        # Under-experienced — partial credit proportional to gap
        ratio = candidate_years / max(min_req, 1)
        return round(min(80.0, ratio * 100), 2)

    if candidate_years > max_req + 3:
        # Significantly overqualified
        return 85.0

    # Within range (or slightly above max)
    return 100.0


def score_education(
    candidate_education: Optional[str],
    job_education:       Optional[str],
) -> float:
    """
    Education score (0–100).
    Simple tier matching — higher degree always scores well.
    """
    if not job_education:
        return 80.0   # no requirement

    if not candidate_education:
        return 40.0   # unknown

    # Education tier map — higher index = higher qualification
    tiers = [
        ["diploma", "certificate", "intermediate", "mpc"],
        ["bachelor", "b.tech", "btech", "b.e", "b.sc", "bca", "engineering"],
        ["master", "m.tech", "mtech", "m.s", "m.sc", "mca", "mba"],
        ["phd", "ph.d", "doctorate"],
    ]

    def get_tier(text: str) -> int:
        t = text.lower()
        for i, tier in enumerate(tiers):
            if any(keyword in t for keyword in tier):
                return i
        return 1   # assume bachelor-equivalent if unknown

    candidate_tier = get_tier(candidate_education)
    required_tier  = get_tier(job_education)

    if candidate_tier >= required_tier:
        return 100.0
    elif candidate_tier == required_tier - 1:
        return 70.0
    else:
        return 40.0


def score_certifications(
    candidate_certs: list[str],
    required_skills: list[str],
) -> float:
    """
    Certification score (0–100).
    Bonus points if certifications complement required skills.
    """
    if not candidate_certs:
        return 80.0   # neutral — not penalised for missing certs

    req_lower = " ".join(required_skills).lower()
    relevant = sum(
        1 for cert in candidate_certs
        if any(word in req_lower for word in cert.lower().split())
    )

    if not relevant:
        return 60.0
    return min(100.0, 60 + relevant * 20)


# ─────────────────────────────────────────────────────────────────
#  Main scorer
# ─────────────────────────────────────────────────────────────────

def compute_score(
    candidate:        Candidate,
    job:              Job,
    semantic_score:   float,   # FAISS cosine similarity 0–1
) -> ScoreBreakdown:
    """
    Compute weighted final score for one candidate against one job.
    """
    skill_score = score_skills(
        candidate.skills or [],
        job.required_skills or [],
        job.nice_to_have_skills or [],
    )
    exp_score = score_experience(
        candidate.experience_years,
        job.experience_min,
        job.experience_max,
    )
    edu_score = score_education(
        candidate.education,
        job.education,
    )
    cert_score = score_certifications(
        candidate.certifications or [],
        job.required_skills or [],
    )

    # Semantic score from FAISS (0–1 → 0–100)
    sem_score_pct = round(semantic_score * 100, 2)

    # Blend keyword skill score + semantic similarity for robustness
    blended_skill = skill_score * 0.6 + sem_score_pct * 0.4

    final = round(
        blended_skill        * WEIGHTS["skill"]
        + exp_score          * WEIGHTS["experience"]
        + edu_score          * WEIGHTS["education"]
        + cert_score         * WEIGHTS["cert"],
        2,
    )

    return ScoreBreakdown(
        skill_score      = round(skill_score, 2),
        experience_score = round(exp_score, 2),
        education_score  = round(edu_score, 2),
        cert_score       = round(cert_score, 2),
        semantic_score   = sem_score_pct,
        final_score      = final,
    )