"""
app/models/user.py
-------------------
SQLAlchemy ORM model for the `users` table.
Stores recruiter accounts with hashed passwords.
"""

import uuid
from datetime import datetime
from sqlalchemy import String, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    email:           Mapped[str]      = mapped_column(String(255), unique=True, nullable=False, index=True)
    full_name:       Mapped[str]      = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[str]      = mapped_column(String(255), nullable=False)
    is_active:       Mapped[bool]     = mapped_column(Boolean, default=True)
    created_at:      Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<User email={self.email}>"
