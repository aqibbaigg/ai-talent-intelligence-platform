"""
app/services/rag_service.py
-----------------------------
RAG (Retrieval-Augmented Generation) pipeline.

Flow:
  Question (natural language)
    ↓
  Embed question → 384-dim vector
    ↓
  FAISS search → top-k relevant resume chunks
    ↓
  Fetch full candidate profiles from DB
    ↓
  Build context string (resume summaries)
    ↓
  GPT-4o prompt: context + question → answer
    ↓
  Return answer + source candidates

What makes this powerful:
  Recruiters can ask questions like:
  "Find candidates with NLP experience and AWS certification"
  "Who has worked at a startup and knows React?"
  "Which candidates have both Python and SQL skills?"

  The LLM understands intent — not just keyword matching.

Week 3 features built here:
  - Module 8: RAG pipeline
  - Module 7: LLM recommendation for individual candidates
"""

import json
from typing import Optional

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.models.candidate import Candidate
from app.models.job import Job
from app.models.match import Match
from app.services import embedder
from app.services.faiss_index import get_faiss_index


# ─────────────────────────────────────────────────────────────────
#  OpenAI client (lazy init)
# ─────────────────────────────────────────────────────────────────

_openai_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        if not settings.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Add it to your .env file."
            )
        try:
            from openai import AsyncOpenAI
            _openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        except ImportError:
            raise ImportError("Run: pip install openai")
    return _openai_client


# ─────────────────────────────────────────────────────────────────
#  Candidate context builder
# ─────────────────────────────────────────────────────────────────

def build_candidate_context(candidates: list[Candidate]) -> str:
    """
    Formats candidate profiles into a context string for the LLM.
    Concise but complete — fits within token limits.
    """
    parts = []
    for i, c in enumerate(candidates, 1):
        skills_str = ", ".join(c.skills[:15]) if c.skills else "Not listed"
        exp_str = f"{c.experience_years} years" if c.experience_years else "Unknown"
        certs_str = ", ".join(c.certifications[:5]) if c.certifications else "None"

        parts.append(f"""
Candidate {i}: {c.name}
  Email:          {c.email or 'N/A'}
  Skills:         {skills_str}
  Experience:     {exp_str}
  Education:      {c.education or 'Unknown'}
  Certifications: {certs_str}
""".strip())

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
    RAG pipeline: embed question → FAISS search → GPT-4o answer.

    Parameters
    ----------
    question : recruiter's natural language query
    db       : async DB session
    top_k    : number of candidates to use as RAG context

    Returns
    -------
    dict with keys: answer, sources, model_used
    """

    # ── Step 1: Embed the question ────────────────────────────────
    question_embedding = await embedder.generate_embedding_async(question)

    # ── Step 2: FAISS search for relevant candidates ──────────────
    index = get_faiss_index()
    if not index.is_ready or index.total == 0:
        return {
            "answer": "No candidates in the system yet. Upload some resumes first.",
            "sources": [],
            "model_used": "none",
        }

    raw_results = index.search(question_embedding, top_k=top_k)
    candidate_ids = [cid for cid, _ in raw_results]

    if not candidate_ids:
        return {
            "answer": "No matching candidates found for your query.",
            "sources": [],
            "model_used": "none",
        }

    # ── Step 3: Fetch candidates from DB ─────────────────────────
    result = await db.execute(
        select(Candidate).where(Candidate.id.in_(candidate_ids))
    )
    candidates = result.scalars().all()

    # ── Step 4: Build context ─────────────────────────────────────
    context = build_candidate_context(list(candidates))

    # ── Step 5: GPT-4o generates answer ──────────────────────────
    system_prompt = """You are an expert recruiter assistant.
You have access to candidate profiles retrieved from a talent database.
Answer the recruiter's question based ONLY on the provided candidate data.
Be specific, mention candidate names, and highlight relevant skills.
If no candidates match the criteria well, say so honestly.
Keep answers concise (3-5 sentences per candidate mentioned)."""

    user_prompt = f"""Recruiter Question: {question}

Retrieved Candidate Profiles:
{context}

Please answer the recruiter's question based on these profiles."""

    try:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,        # low temp = factual, not creative
            max_tokens=800,
        )
        answer = response.choices[0].message.content
        model_used = "gpt-4o"
        logger.info("RAG answer generated — {} chars", len(answer))

    except Exception as e:
        logger.error("LLM call failed: {}", e)
        # Graceful fallback — return structured summary without LLM
        answer = _fallback_answer(question, candidates)
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
    """Template-based fallback when LLM is unavailable."""
    names = [c.name for c in candidates[:5]]
    skills_union = set()
    for c in candidates:
        skills_union.update(c.skills or [])
    return (
        f"Based on your query '{question}', the most relevant candidates are: "
        f"{', '.join(names)}. "
        f"Combined skill coverage includes: {', '.join(list(skills_union)[:10])}."
    )


# ─────────────────────────────────────────────────────────────────
#  Module 7: LLM recommendation for a single candidate
# ─────────────────────────────────────────────────────────────────

async def generate_candidate_recommendation(
    candidate_id: str,
    job_id:       str,
    db:           AsyncSession,
) -> str:
    """
    Generate a recruiter-friendly LLM summary for one candidate.

    "Why should I hire this candidate for this job?"

    Called after matching — enriches Match.llm_summary.
    """

    # Fetch candidate and job
    cand_result = await db.execute(
        select(Candidate).where(Candidate.id == candidate_id)
    )
    candidate = cand_result.scalar_one_or_none()

    job_result = await db.execute(select(Job).where(Job.id == job_id))
    job = job_result.scalar_one_or_none()

    if not candidate or not job:
        return "Candidate or job not found."

    # Fetch match score
    match_result = await db.execute(
        select(Match).where(
            Match.candidate_id == candidate_id,
            Match.job_id == job_id,
        )
    )
    match = match_result.scalar_one_or_none()
    score_info = f"Match score: {match.final_score:.1f}/100" if match else ""

    skills_str = ", ".join(candidate.skills[:20]) if candidate.skills else "Not listed"
    req_skills = ", ".join(job.required_skills) if job.required_skills else "Not specified"

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

Write 3-4 sentences explaining candidate fit. Start with candidate name."""

    try:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=200,
        )
        summary = response.choices[0].message.content.strip()

        # Save to DB
        if match:
            match.llm_summary = summary
            await db.flush()

        return summary

    except Exception as e:
        logger.error("LLM recommendation failed: {}", e)
        # Template fallback
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
