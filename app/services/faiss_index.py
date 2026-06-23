"""
app/services/faiss_index.py
-----------------------------
FAISS index manager for the talent platform.

Architecture
------------
  - One index stores ALL candidate embeddings
  - Index maps FAISS internal integer IDs → candidate UUIDs
  - Rebuilt from DB on startup (fast — ~1s for 10k candidates)
  - Updated incrementally as new resumes are uploaded
"""

import asyncio
import threading
from typing import Optional

import faiss
import numpy as np
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.models.candidate import Candidate


class FAISSIndex:
    """Thread-safe FAISS index wrapper."""

    def __init__(self, dim: int = 384):
        self.dim    = dim
        self._index = faiss.IndexFlatIP(dim)
        self._id_map: list[str] = []
        self._lock  = threading.Lock()
        self._ready = False

    def add(self, candidate_id: str, embedding: list[float]) -> None:
        vec = np.array([embedding], dtype=np.float32)
        with self._lock:
            self._index.add(vec)
            self._id_map.append(candidate_id)

    def add_batch(
        self,
        candidate_ids: list[str],
        embeddings:    list[list[float]],
    ) -> None:
        if not embeddings:
            return
        matrix = np.array(embeddings, dtype=np.float32)
        with self._lock:
            self._index.add(matrix)
            self._id_map.extend(candidate_ids)
        logger.debug("FAISS: added {} vectors, total={}", len(embeddings), self.total)

    def reset(self) -> None:
        """
        Wipe the index completely — used when all candidates are deleted.
        Creates a brand new empty IndexFlatIP so new uploads are indexed fresh.
        """
        with self._lock:
            self._index = faiss.IndexFlatIP(self.dim)
            self._id_map = []
            self._ready  = True
        logger.info("FAISS index reset — 0 candidates indexed")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        if self._index.ntotal == 0:
            logger.warning("FAISS index is empty — no candidates indexed yet")
            return []

        k     = min(top_k, self._index.ntotal)
        query = np.array([query_embedding], dtype=np.float32)

        with self._lock:
            distances, indices = self._index.search(query, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            candidate_uuid = self._id_map[idx]
            if candidate_uuid == "__deleted__":
                continue
            similarity = float(max(0.0, dist))
            results.append((candidate_uuid, similarity))

        return results

    def remove(self, candidate_id: str) -> None:
        with self._lock:
            for i, cid in enumerate(self._id_map):
                if cid == candidate_id:
                    self._id_map[i] = "__deleted__"
                    break

    @property
    def total(self) -> int:
        return self._index.ntotal

    @property
    def is_ready(self) -> bool:
        return self._ready


# ─────────────────────────────────────────────────────────────────
#  Global singleton
# ─────────────────────────────────────────────────────────────────

_faiss_index: Optional[FAISSIndex] = None


def get_faiss_index() -> FAISSIndex:
    global _faiss_index
    if _faiss_index is None:
        _faiss_index = FAISSIndex(dim=settings.EMBEDDING_DIM)
    return _faiss_index


# ─────────────────────────────────────────────────────────────────
#  Startup loader
# ─────────────────────────────────────────────────────────────────

async def build_index_from_db(db: AsyncSession) -> FAISSIndex:
    global _faiss_index
    index = get_faiss_index()

    logger.info("Building FAISS index from database...")

    result = await db.execute(
        select(Candidate.id, Candidate.embedding)
        .where(Candidate.embedding.is_not(None))
    )
    rows = result.all()

    if not rows:
        logger.warning("No candidate embeddings found in DB — index is empty")
        index._ready = True
        return index

    candidate_ids = [row.id for row in rows]
    embeddings    = [row.embedding for row in rows]

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, index.add_batch, candidate_ids, embeddings)

    index._ready = True
    logger.info("FAISS index ready — {} candidates indexed", index.total)
    _faiss_index = index
    return index