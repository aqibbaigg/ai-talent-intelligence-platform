"""
app/services/auth_service.py
------------------------------
JWT authentication service.

Handles:
  - Password hashing and verification (bcrypt)
  - JWT token creation and validation
  - User registration and login
"""

from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger
from app.core.config import settings
from app.models.user import User

# ── Password hashing ──────────────────────────────────────────────
pwd_context = CryptContext(
    schemes=["bcrypt_sha256", "bcrypt"],
    deprecated="auto"
)

# ── JWT config ────────────────────────────────────────────────────
ALGORITHM       = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


# ── User operations ───────────────────────────────────────────────

async def register_user(
    email:     str,
    full_name: str,
    password:  str,
    db:        AsyncSession,
) -> User:
    # Check if email already exists
    result = await db.execute(select(User).where(User.email == email.lower()))
    if result.scalar_one_or_none():
        raise ValueError("Email already registered.")

    user = User(
        id              = __import__('uuid').uuid4().__str__(),
        email           = email.lower(),
        full_name       = full_name,
        hashed_password = hash_password(password),
    )
    db.add(user)
    await db.flush()
    logger.info("New user registered — email={}", email)
    return user


async def login_user(
    email:    str,
    password: str,
    db:       AsyncSession,
) -> tuple[User, str]:
    result = await db.execute(select(User).where(User.email == email.lower()))
    user   = result.scalar_one_or_none()

    if not user or not verify_password(password, user.hashed_password):
        raise ValueError("Invalid email or password.")

    if not user.is_active:
        raise ValueError("Account is disabled.")

    token = create_access_token({"sub": user.id, "email": user.email})
    logger.info("User logged in — email={}", email)
    return user, token


async def get_current_user(token: str, db: AsyncSession) -> Optional[User]:
    payload = decode_token(token)
    if not payload:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()
