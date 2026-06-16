"""
main.py  —  AI Talent Intelligence Platform
---------------------------------------------
Startup sequence:
  1. Load config from .env
  2. Init DB (create tables, enable pgvector)
  3. Build FAISS index from existing candidates in DB
  4. Warm up embedding model
  5. Register routers: resume, jobs, chat
  6. Serve
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import settings
from app.core.database import init_db, AsyncSessionLocal
from app.services.embedder import get_model
from app.services.faiss_index import build_index_from_db
from app.api.resume import router as resume_router
from app.api.jobs   import router as jobs_router
from app.api.chat   import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting {} v{}", settings.APP_NAME, settings.APP_VERSION)

    # 1. Init DB
    await init_db()

    # 2. Warm up embedding model
    logger.info("Loading embedding model: {}", settings.EMBEDDING_MODEL)
    get_model()

    # 3. Build FAISS index from DB
    async with AsyncSessionLocal() as db:
        await build_index_from_db(db)

    logger.info("All systems ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
## AI Talent Intelligence Platform

### Week 1 — Resume ingestion
- **POST /api/v1/upload-resume** — parse PDF, extract skills, store 384-dim embedding

### Week 2 — FAISS matching
- **POST /api/v1/jobs**   — create job with embedding
- **POST /api/v1/match**  — FAISS search + weighted scoring
- **GET  /api/v1/matches/{job_id}** — retrieve saved results

### Week 3 — LLM / RAG
- **POST /api/v1/chat**      — RAG Q&A over all resumes
- **POST /api/v1/recommend** — GPT-4o candidate summary
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(resume_router)
app.include_router(jobs_router)
app.include_router(chat_router)


@app.get("/health", tags=["system"])
async def health():
    from app.services.faiss_index import get_faiss_index
    idx = get_faiss_index()
    return {
        "status":              "ok",
        "version":             settings.APP_VERSION,
        "embedding_model":     settings.EMBEDDING_MODEL,
        "candidates_indexed":  idx.total,
        "faiss_ready":         idx.is_ready,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000,
                reload=settings.DEBUG, log_level="info")
