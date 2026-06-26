"""
main.py  —  AI Talent Intelligence Platform
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import os
from app.core.config import settings
from app.core.database import init_db, AsyncSessionLocal
from app.services.embedder import get_model
from app.services.faiss_index import build_index_from_db
from app.api.resume import router as resume_router
from app.api.jobs   import router as jobs_router
from app.api.chat   import router as chat_router
from app.api.auth   import router as auth_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting {} v{}", settings.APP_NAME, settings.APP_VERSION)
    from app.models import candidate, job, match, user  # noqa
    await init_db()
    logger.info("Loading embedding model: {}", settings.EMBEDDING_MODEL)
    get_model()
    async with AsyncSessionLocal() as db:
        await build_index_from_db(db)
    logger.info("All systems ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Manual CORS middleware ────────────────────────────────────────
@app.middleware("http")
async def cors_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        response = JSONResponse(content={}, status_code=200)
        response.headers["Access-Control-Allow-Origin"]  = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Max-Age"]       = "3600"
        return response
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(resume_router)
app.include_router(jobs_router)
app.include_router(chat_router)


# ── Serve frontend static files ───────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_login():
    return FileResponse("static/login.html")

@app.get("/app", include_in_schema=False)
async def serve_app():
    return FileResponse("static/index.html")

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health", tags=["system"])
async def health():
    from app.services.faiss_index import get_faiss_index
    idx = get_faiss_index()
    return {
        "status":             "ok",
        "version":            settings.APP_VERSION,
        "embedding_model":    settings.EMBEDDING_MODEL,
        "candidates_indexed": idx.total,
        "faiss_ready":        idx.is_ready,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000,
                reload=settings.DEBUG, log_level="info")