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

Why rebuild on startup instead of persisting to disk?
  - Avoids stale index (DB is always source of truth)
  - 10k candidates rebuild in ~0.5s — negligible startup cost
  - Persisting adds complexity without benefit at this scale
  - Switch to disk persistence when you exceed 100k candidates

Index type: IndexFlatIP (inner product = cosine on L2-normalised vectors)
  - Exact search (no approximation)
  - Perfect recall
  - Switch to IndexIVFFlat for 500k+ candidates
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


# ─────────────────────────────────────────────────────────────────
#  FAISSIndex singleton
# ─────────────────────────────────────────────────────────────────

class FAISSIndex:
    """
    Thread-safe FAISS index wrapper.

    Internal ID mapping:
      FAISS assigns integer IDs (0, 1, 2 …) as vectors are added.
      We keep a parallel list `_id_map` where _id_map[faiss_int] = candidate_uuid.
      This lets us translate FAISS results back to UUIDs.
    """

    def __init__(self, dim: int = 384):
        self.dim    = dim
        self._index = faiss.IndexFlatIP(dim)   # Inner Product (cosine for L2-normed)
        self._id_map: list[str] = []            # faiss_int → candidate_uuid
        self._lock  = threading.Lock()
        self._ready = False

    # ------------------------------------------------------------------
    def add(self, candidate_id: str, embedding: list[float]) -> None:
        """
        Add one candidate embedding to the index.
        Thread-safe — called from async context via run_in_executor.
        """
        vec = np.array([embedding], dtype=np.float32)   # shape (1, 384)
        with self._lock:
            self._index.add(vec)
            self._id_map.append(candidate_id)

    def add_batch(
        self,
        candidate_ids: list[str],
        embeddings:    list[list[float]],
    ) -> None:
        """
        Add multiple embeddings at once — much faster than individual add().
        Used when rebuilding the index from DB on startup.
        """
        if not embeddings:
            return
        matrix = np.array(embeddings, dtype=np.float32)  # (N, 384)
        with self._lock:
            self._index.add(matrix)
            self._id_map.extend(candidate_ids)
        logger.debug("FAISS: added {} vectors, total={}", len(embeddings), self.total)

    # ------------------------------------------------------------------
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Find top-k most similar candidates.

        Parameters
        ----------
        query_embedding : 384-dim float list (L2-normalised)
        top_k           : number of results

        Returns
        -------
        list of (candidate_uuid, similarity_score)
        similarity_score is in [0, 1] — higher is better
        """
        if self._index.ntotal == 0:
            logger.warning("FAISS index is empty — no candidates indexed yet")
            return []

        k = min(top_k, self._index.ntotal)
        query = np.array([query_embedding], dtype=np.float32)  # (1, 384)

        with self._lock:
            distances, indices = self._index.search(query, k)  # (1, k), (1, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            candidate_uuid = self._id_map[idx]
            # IndexFlatIP returns cosine similarity directly (already in [-1, 1])
            # We clamp to [0, 1] and convert to percentage
            similarity = float(max(0.0, dist))
            results.append((candidate_uuid, similarity))

        return results

    def remove(self, candidate_id: str) -> None:
        """
        Mark a candidate as removed.
        True FAISS removal requires index rebuild — we mark as tombstone.
        Index is rebuilt on next startup anyway.
        """
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
#  Global singleton instance
# ─────────────────────────────────────────────────────────────────

_faiss_index: Optional[FAISSIndex] = None


def get_faiss_index() -> FAISSIndex:
    """FastAPI dependency — returns the global FAISS index."""
    global _faiss_index
    if _faiss_index is None:
        _faiss_index = FAISSIndex(dim=settings.EMBEDDING_DIM)
    return _faiss_index


# ─────────────────────────────────────────────────────────────────
#  Startup loader — rebuilds index from PostgreSQL
# ─────────────────────────────────────────────────────────────────

async def build_index_from_db(db: AsyncSession) -> FAISSIndex:
    """
    Load all candidate embeddings from PostgreSQL and build the FAISS index.

    Called once at app startup via lifespan().
    For 10,000 candidates this takes ~0.5 seconds.

    Parameters
    ----------
    db : async DB session

    Returns
    -------
    FAISSIndex — populated and ready for search
    """
    global _faiss_index
    index = get_faiss_index()

    logger.info("Building FAISS index from database...")

    # Fetch all candidates that have embeddings
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

    # Build in thread pool (numpy operations, not async)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        index.add_batch,
        candidate_ids,
        embeddings,
    )

    index._ready = True
    logger.info("FAISS index ready — {} candidates indexed", index.total)
    _faiss_index = index
    return index
