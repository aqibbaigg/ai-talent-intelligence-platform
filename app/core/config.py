"""
app/core/config.py
------------------
Central configuration. All settings come from .env file.
Never hardcode secrets — always use environment variables.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────────────
    APP_NAME: str = "AI Talent Intelligence Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    SECRET_KEY: str = "change-in-production"

    # ── Database ──────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/talent_db"
    DATABASE_SYNC_URL: str = "postgresql://postgres:password@localhost:5432/talent_db"

    # ── Embedding ─────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # ── File upload ───────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 10
    UPLOAD_DIR: str = "uploads"

    # ── LLM (Week 3) ─────────────────────────────────────────────
    OPENAI_API_KEY: str = ""

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        p = Path(self.UPLOAD_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
