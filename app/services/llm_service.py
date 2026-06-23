"""
app/services/llm_service.py
-----------------------------
Central LLM service using Ollama (llama3.2).

All LLM calls go through this module so prompts are consistent
and switching models only requires changing OLLAMA_MODEL.

Functions:
  generate_text(prompt)                        → str
  generate_candidate_summary(candidate, job)   → str
  generate_interview_questions(candidate, job) → list[str]
  generate_job_summary(job)                    → dict
"""

import asyncio
import json
import urllib.request
import urllib.error
from loguru import logger

from app.models.candidate import Candidate
from app.models.job import Job

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


# ─────────────────────────────────────────────────────────────────
#  Core Ollama caller
# ─────────────────────────────────────────────────────────────────

def _call_ollama_sync(prompt: str, max_tokens: int = 500) -> str:
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.3,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip()
    except urllib.error.URLError as e:
        raise RuntimeError(
            "Ollama not reachable. Make sure it is running."
        ) from e


async def generate_text(prompt: str, max_tokens: int = 500) -> str:
    """Generate text from any prompt using Ollama."""
    return await asyncio.get_event_loop().run_in_executor(
        None, lambda: _call_ollama_sync(prompt, max_tokens)
    )


# ─────────────────────────────────────────────────────────────────
#  Candidate summary
# ─────────────────────────────────────────────────────────────────

async def generate_candidate_summary(
    candidate: Candidate,
    job: Job | None = None,
) -> str:
    """
    Generate a recruiter-friendly summary for a candidate.
    If a job is provided, the summary focuses on fit for that role.
    """
    skills_str = ", ".join(candidate.skills[:20]) if candidate.skills else "Not listed"
    exp_str    = f"{candidate.experience_years} years" if candidate.experience_years else "Unknown"
    certs_str  = ", ".join(candidate.certifications) if candidate.certifications else "None"

    if job:
        req_skills = ", ".join(job.required_skills) if job.required_skills else "Not specified"
        job_context = f"""
Target Job: {job.title}
Required Skills: {req_skills}
Experience Required: {job.experience_min or 0}–{job.experience_max or 'any'} years"""
    else:
        job_context = ""

    prompt = f"""You are a senior recruiter. Write a concise 3-4 sentence professional summary
of this candidate for a recruiter. Focus on their strengths and key skills.
{f"Assess their fit for the job below." if job else ""}

Candidate: {candidate.name}
Skills: {skills_str}
Experience: {exp_str}
Education: {candidate.education or 'Unknown'}
Certifications: {certs_str}
{job_context}

Write 3-4 sentences. Start with the candidate's name. Be specific and professional.

Summary:"""

    try:
        return await generate_text(prompt, max_tokens=250)
    except Exception as e:
        logger.error("generate_candidate_summary failed: {}", e)
        matched = []
        if job:
            matched = [s for s in (candidate.skills or []) if s in (job.required_skills or [])]
        return (
            f"{candidate.name} has {exp_str} of experience with skills including "
            f"{', '.join((candidate.skills or [])[:5])}. "
            + (f"Matches {len(matched)} of {len(job.required_skills)} required skills." if job else "")
        )


# ─────────────────────────────────────────────────────────────────
#  Interview questions
# ─────────────────────────────────────────────────────────────────

async def generate_interview_questions(
    candidate: Candidate,
    job: Job | None = None,
) -> list[str]:
    """
    Generate 5 tailored interview questions based on candidate skills.
    If a job is provided, questions focus on role-specific gaps and strengths.
    """
    skills_str = ", ".join(candidate.skills[:15]) if candidate.skills else "Not listed"
    exp_str    = f"{candidate.experience_years} years" if candidate.experience_years else "Unknown"

    if job:
        req_skills  = ", ".join(job.required_skills) if job.required_skills else "Not specified"
        job_context = f"Job Title: {job.title}\nRequired Skills: {req_skills}"
    else:
        job_context = ""

    prompt = f"""You are a senior technical recruiter. Generate exactly 5 tailored interview questions
for this candidate. Questions should be specific to their skills and experience level.
Mix technical and behavioral questions. Return ONLY the 5 questions as a numbered list.

Candidate: {candidate.name}
Skills: {skills_str}
Experience: {exp_str}
Education: {candidate.education or 'Unknown'}
{job_context}

5 Interview Questions:
1."""

    try:
        raw = await generate_text(prompt, max_tokens=400)
        # Parse numbered list from response
        lines = raw.strip().splitlines()
        questions = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove numbering like "1.", "2.", "1)", etc.
            import re
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if cleaned and len(cleaned) > 10:
                questions.append(cleaned)
        # Ensure we always return exactly 5
        if len(questions) >= 5:
            return questions[:5]
        # Fallback questions if parsing failed
        return questions + _fallback_questions(candidate, job)[len(questions):5]

    except Exception as e:
        logger.error("generate_interview_questions failed: {}", e)
        return _fallback_questions(candidate, job)


def _fallback_questions(candidate: Candidate, job: Job | None) -> list[str]:
    skills = candidate.skills or []
    top_skills = skills[:3] if skills else ["your primary skills"]
    role = job.title if job else "this role"
    return [
        f"Can you walk me through a project where you used {top_skills[0]}?",
        f"How do you approach debugging complex issues in {top_skills[1] if len(top_skills) > 1 else 'your work'}?",
        f"What makes you a strong fit for the {role} position?",
        "Describe a situation where you had to learn a new technology quickly. How did you approach it?",
        "How do you ensure code quality and maintainability in your projects?",
    ]


# ─────────────────────────────────────────────────────────────────
#  Job summary
# ─────────────────────────────────────────────────────────────────

async def generate_job_summary(job: Job) -> dict:
    """
    Generate a structured summary of job requirements.
    Returns summary text + must_have_skills + nice_to_have_skills.
    """
    req_skills  = ", ".join(job.required_skills)    if job.required_skills    else "Not specified"
    nice_skills = ", ".join(job.nice_to_have_skills) if job.nice_to_have_skills else "None"

    prompt = f"""You are a recruiter. Summarize this job posting in 2-3 sentences.
Then list must-have skills and nice-to-have skills as comma-separated values.

Job Title: {job.title}
Company: {job.company or 'Not specified'}
Description: {job.description[:500]}
Required Skills: {req_skills}
Nice to Have: {nice_skills}
Experience: {job.experience_min or 0}–{job.experience_max or 'any'} years
Location: {job.location or 'Not specified'}
Job Type: {job.job_type or 'Not specified'}

Respond in this exact format:
SUMMARY: <2-3 sentence summary>
MUST_HAVE: <comma separated must-have skills>
NICE_TO_HAVE: <comma separated nice-to-have skills>"""

    try:
        raw = await generate_text(prompt, max_tokens=300)

        # Parse structured response
        summary      = ""
        must_have    = []
        nice_to_have = []

        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
            elif line.startswith("MUST_HAVE:"):
                raw_skills = line.replace("MUST_HAVE:", "").strip()
                must_have  = [s.strip() for s in raw_skills.split(",") if s.strip()]
            elif line.startswith("NICE_TO_HAVE:"):
                raw_skills   = line.replace("NICE_TO_HAVE:", "").strip()
                nice_to_have = [s.strip() for s in raw_skills.split(",") if s.strip()]

        # Fallback to job data if parsing failed
        if not summary:
            summary = f"This {job.title} role requires {req_skills}."
        if not must_have:
            must_have = job.required_skills or []
        if not nice_to_have:
            nice_to_have = job.nice_to_have_skills or []

        return {
            "summary":           summary,
            "must_have_skills":  must_have,
            "nice_to_have_skills": nice_to_have,
        }

    except Exception as e:
        logger.error("generate_job_summary failed: {}", e)
        return {
            "summary":             f"This {job.title} role requires {req_skills}.",
            "must_have_skills":    job.required_skills or [],
            "nice_to_have_skills": job.nice_to_have_skills or [],
        }
