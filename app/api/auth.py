"""
app/api/auth.py
----------------
Authentication endpoints.

POST /api/v1/auth/register   Create recruiter account
POST /api/v1/auth/login      Login and get JWT token
GET  /api/v1/auth/me         Get current user info
"""

from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from pydantic import BaseModel, EmailStr, Field
from loguru import logger

from app.core.database import get_db
from app.services.auth_service import register_user, login_user, get_current_user

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


# ── Schemas ───────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str = Field(..., description="Recruiter email")
    full_name: str = Field(..., min_length=2)
    password: str = Field(..., min_length=6, max_length=64)


class LoginRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=6, max_length=64)


class AuthResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user_id:      str
    email:        str
    full_name:    str


# ── Endpoints ─────────────────────────────────────────────────────

@router.post("/register", response_model=AuthResponse, status_code=201)
async def register(
    data: RegisterRequest,
    db:   AsyncSession = Depends(get_db),
) -> AuthResponse:
    try:
        user, token = await _register_and_login(data, db)
        return AuthResponse(
            access_token = token,
            user_id      = user.id,
            email        = user.email,
            full_name    = user.full_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Registration failed: {}", e)
        raise HTTPException(status_code=500, detail="Registration failed")


async def _register_and_login(data: RegisterRequest, db: AsyncSession):
    from app.services.auth_service import create_access_token
    user  = await register_user(data.email, data.full_name, data.password, db)
    token = create_access_token({"sub": user.id, "email": user.email})
    return user, token


@router.post("/login", response_model=AuthResponse)
async def login(
    data: LoginRequest,
    db:   AsyncSession = Depends(get_db),
) -> AuthResponse:
    try:
        user, token = await login_user(data.email, data.password, db)
        return AuthResponse(
            access_token = token,
            user_id      = user.id,
            email        = user.email,
            full_name    = user.full_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error("Login failed: {}", e)
        raise HTTPException(status_code=500, detail="Login failed")


@router.get("/me")
async def get_me(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization.replace("Bearer ", "")
    user  = await get_current_user(token, db)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return {
        "user_id":   user.id,
        "email":     user.email,
        "full_name": user.full_name,
        "is_active": user.is_active,
    }
