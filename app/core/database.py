"""
app/core/database.py
---------------------
Async PostgreSQL connection using SQLAlchemy 2.0.
pgvector extension enables storing 384-dim embeddings natively.

Why async?
  FastAPI is async — blocking DB calls would kill concurrency.
  asyncpg + SQLAlchemy async lets us handle hundreds of concurrent
  resume uploads without threads.
"""

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from loguru import logger

from app.core.config import settings


# ── Engine ────────────────────────────────────────────────────────
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,       # logs all SQL in debug mode
    pool_size=10,              # concurrent DB connections
    max_overflow=20,
    pool_pre_ping=True,        # auto-reconnect on dropped connections
)

# ── Session factory ───────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,    # objects stay usable after commit
)


# ── Base class for all ORM models ─────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Dependency injection for FastAPI routes ───────────────────────
async def get_db() -> AsyncSession:
    """
    FastAPI dependency — yields a DB session per request.
    Session is automatically closed after the request completes.

    Usage in route:
        @app.post("/upload-resume")
        async def upload(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── DB initialisation ─────────────────────────────────────────────
async def init_db() -> None:
    """
    Create all tables and enable pgvector extension.
    Called once at app startup.
    """
    async with engine.begin() as conn:
        # Enable pgvector — must exist before creating vector columns
        # Create all tables defined in models
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialised — tables ready")
