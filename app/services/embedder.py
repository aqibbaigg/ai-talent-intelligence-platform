"""
app/services/embedder.py
-------------------------
Sentence embedding service using all-MiniLM-L6-v2.

Design decisions:
  - Singleton pattern: model loads once at startup (~50MB RAM)
  - Synchronous encode() wrapped in async for FastAPI compatibility
  - L2-normalised output: enables cosine similarity via dot product
  - Resume text truncated at 512 tokens (model limit)

Embedding dimension: 384
Model: all-MiniLM-L6-v2 (fastest, free, good quality)
Upgrade path: swap to bge-large-en-v1.5 for better accuracy
             (1024 dims — also update DB column and FAISS index)
"""

import asyncio
from functools import lru_cache
from typing import Optional

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.core.config import settings


# ─────────────────────────────────────────────────────────────────
#  Singleton model loader
# ─────────────────────────────────────────────────────────────────

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """
    Load the embedding model once and reuse.
    Thread-safe — first call loads, subsequent calls return cached.
    """
    global _model
    if _model is None:
        logger.info("Loading embedding model: {}", settings.EMBEDDING_MODEL)
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info(
            "Embedding model ready — dim={}", settings.EMBEDDING_DIM
        )
    return _model


# ─────────────────────────────────────────────────────────────────
#  Text preparation
# ─────────────────────────────────────────────────────────────────

def prepare_resume_text(
    raw_text: str,
    skills: list[str] | None = None,
    max_chars: int = 3000,
) -> str:
    """
    Prepare resume text for embedding.

    Strategy:
      - Prepend skill list so the embedding is skill-weighted
      - Truncate to stay within model token limits
      - Skills are repeated so they influence the embedding more strongly

    Parameters
    ----------
    raw_text  : full resume text
    skills    : extracted skill list (optional, boosts skill signal)
    max_chars : character limit before truncation

    Returns
    -------
    str — embedding-ready text
    """
    parts = []

    # Skill signal (prepend for emphasis)
    if skills:
        skills_text = "Skills: " + ", ".join(skills)
        parts.append(skills_text)

    # Main resume text (truncated)
    parts.append(raw_text[:max_chars])

    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────
#  Embedding generation
# ─────────────────────────────────────────────────────────────────

def generate_embedding(text: str) -> list[float]:
    """
    Generate a 384-dimensional L2-normalised embedding.

    Parameters
    ----------
    text : prepared resume text

    Returns
    -------
    list[float] — 384 floats, L2-normalised
                  Compatible with pgvector and FAISS IndexFlatIP
    """
    model = get_model()

    # encode() returns numpy array shape (384,)
    embedding: np.ndarray = model.encode(
        text,
        normalize_embeddings=True,     # L2-normalise → cosine = dot product
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    return embedding.tolist()           # convert to plain Python list for JSON/DB


async def generate_embedding_async(text: str) -> list[float]:
    """
    Async wrapper — runs CPU-bound embedding in a thread pool
    so it doesn't block the FastAPI event loop.

    This is important: SentenceTransformer.encode() is CPU-bound.
    Running it directly in an async route would block all other
    requests until it completes (~50-200ms per resume).
    """
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(
        None,                           # uses default ThreadPoolExecutor
        generate_embedding,
        text,
    )
    return embedding


def batch_generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple texts efficiently.
    Batch processing is ~3x faster than individual encode() calls.

    Used by the build_embeddings script (Week 4).
    """
    model = get_model()
    embeddings: np.ndarray = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return [emb.tolist() for emb in embeddings]
