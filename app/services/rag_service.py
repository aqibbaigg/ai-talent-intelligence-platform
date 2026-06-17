"""
app/services/rag_service.py
-----------------------------
RAG (Retrieval-Augmented Generation) pipeline.

LLM: Ollama (llama3.2) — runs locally, no API key needed, no limits.

Flow:
  Question (natural language)
    ↓
  Embed question → 384-dim vector
    ↓
  FAISS search → top-k relevant candidates
    ↓
  Fetch full candidate profiles from DB
    ↓
  Build context string
    ↓
  Ollama llama3.2 → answer
    ↓
  Return answer + source candidates
"""

import asyncio
import urllib.request
import urllib.error
import json
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.candidate import Candidate
from app.models.job import Job
from app.models.match import Match
from app.services import embedder
from app.services.faiss_index import get_faiss_index

OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


# ─────────────────────────────────────────────────────────────────
#  Ollama call
# ─────────────────────────────────────────────────────────────────

def _call_ollama_sync(prompt: str, max_tokens: int = 500) -> str:
    """
    Call local Ollama API synchronously.
    Ollama runs at http://localhost:11434 after installation.
    """
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
            f"Ollama not reachable at {OLLAMA_URL}. "
            "Make sure Ollama is running: open a terminal and run 'ollama serve'"
        ) from e


async def _call_ollama(prompt: str, max_tokens: int = 500) -> str:
    """Async wrapper — runs sync Ollama call in thread pool."""
    return await asyncio.get_event_loop().run_in_executor(
        None, lambda: _call_ollama_sync(prompt, max_tokens)
    )


# ─────────────────────────────────────────────────────────────────
#  Candidate context builder
# ─────────────────────────────────────────────────────────────────

def build_candidate_context(candidates: list[Candidate]) -> str:
    parts = []
    for i, c in enumerate(candidates, 1):
        skills_str = ", ".join(c.skills[:15]) if c.skills else "Not listed"
        exp_str    = f"{c.experience_years} years" if c.experience_years else "Unknown"
        certs_str  = ", ".join(c.certifications[:5]) if c.certifications else "None"

        parts.append(
            f"Candidate {i}: {c.name}\n"
            f"  Email:          {c.email or 'N/A'}\n"
            f"  Skills:         {skills_str}\n"
            f"  Experience:     {exp_str}\n"
            f"  Education:      {c.education or 'Unknown'}\n"
            f"  Certifications: {certs_str}"
        )
    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────
#  RAG: Answer recruiter questions
# ─────────────────────────────────────────────────────────────────

async def answer_recruiter_question(
    question: str,
    db:       AsyncSession,
    top_k:    int = 5,
) -> dict:
    """
    RAG pipeline: embed question → FAISS search → Ollama answer.
    """

    # ── Step 1: Embed the question ────────────────────────────────
    question_embedding = await embedder.generate_embedding_async(question)

    # ── Step 2: FAISS search ──────────────────────────────────────
    index = get_faiss_index()
    if not index.is_ready or index.total == 0:
        return {
            "answer":     "No candidates in the system yet. Upload some resumes first.",
            "sources":    [],
            "model_used": "none",
        }

    raw_results   = index.search(question_embedding, top_k=top_k)
    candidate_ids = [cid for cid, _ in raw_results]

    if not candidate_ids:
        return {
            "answer":     "No matching candidates found for your query.",
            "sources":    [],
            "model_used": "none",
        }

    # ── Step 3: Fetch candidates from DB ─────────────────────────
    result     = await db.execute(
        select(Candidate).where(Candidate.id.in_(candidate_ids))
    )
    candidates = result.scalars().all()

    # ── Step 4: Build context ─────────────────────────────────────
    context = build_candidate_context(list(candidates))

    # ── Step 5: Ollama generates answer ──────────────────────────
    prompt = f"""You are an expert recruiter assistant.
You have access to candidate profiles retrieved from a talent database.
Answer the recruiter's question based ONLY on the provided candidate data.
Be specific, mention candidate names, and highlight relevant skills.
If no candidates match the criteria well, say so honestly.
Keep your answer concise (3-5 sentences per candidate mentioned).

Recruiter Question: {question}

Retrieved Candidate Profiles:
{context}

Answer:"""

    try:
        answer     = await _call_ollama(prompt, max_tokens=500)
        model_used = OLLAMA_MODEL
        logger.info("RAG answer generated — {} chars", len(answer))

    except Exception as e:
        logger.error("Ollama call failed: {}", e)
        answer     = _fallback_answer(question, list(candidates))
        model_used = "fallback_template"

    sources = [
        {"candidate_id": c.id, "name": c.name, "email": c.email}
        for c in candidates
    ]

    return {
        "answer":     answer,
        "sources":    sources,
        "model_used": model_used,
    }


def _fallback_answer(question: str, candidates: list[Candidate]) -> str:
    names        = [c.name for c in candidates[:5]]
    skills_union = set()
    for c in candidates:
        skills_union.update(c.skills or [])
    return (
        f"Based on your query '{question}', the most relevant candidates are: "
        f"{', '.join(names)}. "
        f"Combined skill coverage includes: {', '.join(list(skills_union)[:10])}."
    )


# ─────────────────────────────────────────────────────────────────
#  LLM recommendation for a single candidate
# ─────────────────────────────────────────────────────────────────

async def generate_candidate_recommendation(
    candidate_id: str,
    job_id:       str,
    db:           AsyncSession,
) -> str:
    """
    Generate a recruiter-friendly LLM summary for one candidate.
    Saves the result to Match.llm_summary in the DB.
    """

    # Fetch candidate
    cand_result = await db.execute(
        select(Candidate).where(Candidate.id == candidate_id)
    )
    candidate = cand_result.scalar_one_or_none()

    # Fetch job
    job_result = await db.execute(select(Job).where(Job.id == job_id))
    job        = job_result.scalar_one_or_none()

    if not candidate or not job:
        return "Candidate or job not found."

    # Fetch match scores
    match_result = await db.execute(
        select(Match).where(
            Match.candidate_id == candidate_id,
            Match.job_id == job_id,
        )
    )
    match      = match_result.scalar_one_or_none()
    score_info = f"Match score: {match.final_score:.1f}/100" if match else ""

    skills_str = ", ".join(candidate.skills[:20]) if candidate.skills else "Not listed"
    req_skills = ", ".join(job.required_skills)   if job.required_skills else "Not specified"

    prompt = f"""You are a senior recruiter. Write a concise recommendation (3-4 sentences)
for whether to hire this candidate for the job. Be specific about skill alignment.

Candidate: {candidate.name}
Skills: {skills_str}
Experience: {candidate.experience_years or 'Unknown'} years
Education: {candidate.education or 'Unknown'}
Certifications: {', '.join(candidate.certifications) if candidate.certifications else 'None'}

Job Title: {job.title}
Required Skills: {req_skills}
Experience Required: {job.experience_min or 0}–{job.experience_max or 'any'} years
{score_info}

Write 3-4 sentences explaining candidate fit. Start with the candidate's name.

Recommendation:"""

    try:
        summary = await _call_ollama(prompt, max_tokens=250)
        summary = summary.strip()

        # Save to DB
        if match:
            match.llm_summary = summary
            await db.flush()

        logger.info(
            "LLM summary generated for candidate={} job={}",
            candidate_id[:8], job_id[:8],
        )
        return summary

    except Exception as e:
        logger.error("Ollama recommendation failed: {}", e)
        matched_skills = [
            s for s in (candidate.skills or [])
            if s in (job.required_skills or [])
        ]
        return (
            f"{candidate.name} matches {len(matched_skills)} of "
            f"{len(job.required_skills)} required skills "
            f"({', '.join(matched_skills[:5])}). "
            f"With {candidate.experience_years or 'unknown'} years experience, "
            f"{'this candidate meets' if matched_skills else 'further review recommended for'} "
            f"the {job.title} role requirements."
        )