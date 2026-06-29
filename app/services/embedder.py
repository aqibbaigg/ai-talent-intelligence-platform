"""
app/services/embedder.py
-------------------------
Sentence embedding service using all-MiniLM-L6-v2.

Railway-safe version:
- Loads model once
- Limits CPU threads
- Truncates long text
- Returns 384-dimensional normalized embeddings
"""

import os
import asyncio
from typing import Optional

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.core.config import settings


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _model

    if _model is None:
        logger.info("Loading embedding model: {}", settings.EMBEDDING_MODEL)

        _model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            device="cpu",
        )

        logger.info(
            "Embedding model ready — dim={}",
            settings.EMBEDDING_DIM,
        )

    return _model


def prepare_resume_text(
    raw_text: str,
    skills: list[str] | None = None,
    max_chars: int = 2500,
) -> str:
    parts = []

    if skills:
        skills_text = "Skills: " + ", ".join(skills)
        parts.append(skills_text)
        parts.append(skills_text)

    clean_text = " ".join((raw_text or "").split())
    parts.append(clean_text[:max_chars])

    return "\n\n".join(parts)


def generate_embedding(text: str) -> list[float]:
    if not text or not text.strip():
        logger.warning("Empty text received for embedding. Returning zero vector.")
        return [0.0] * settings.EMBEDDING_DIM

    model = get_model()

    embedding: np.ndarray = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    embedding_list = embedding.astype(float).tolist()

    if len(embedding_list) != settings.EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: expected {settings.EMBEDDING_DIM}, got {len(embedding_list)}"
        )

    return embedding_list


async def generate_embedding_async(text: str) -> list[float]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, generate_embedding, text)


def batch_generate_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    model = get_model()

    embeddings: np.ndarray = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=8,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    return [emb.astype(float).tolist() for emb in embeddings]