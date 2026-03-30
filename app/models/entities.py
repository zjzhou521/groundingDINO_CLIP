import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Logo(Base):
    __tablename__ = "logo"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_logo_user_name"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    reference_images: Mapped[list["LogoReferenceImage"]] = relationship(back_populates="logo")


class LogoReferenceImage(Base):
    __tablename__ = "logo_reference_images"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    logo_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("logo.id", ondelete="CASCADE"),
        index=True,
    )
    source_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    storage_key: Mapped[str] = mapped_column(String(512), unique=True)
    storage_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    width: Mapped[int] = mapped_column()
    height: Mapped[int] = mapped_column()
    qdrant_point_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    logo: Mapped["Logo"] = relationship(back_populates="reference_images")


class Product(Base):
    __tablename__ = "products"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    source_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    storage_key: Mapped[str] = mapped_column(String(512), unique=True)
    storage_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    width: Mapped[int] = mapped_column()
    height: Mapped[int] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    jobs: Mapped[list["Job"]] = relationship(back_populates="product")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    job_type: Mapped[str] = mapped_column(String(100), index=True)
    status: Mapped[str] = mapped_column(String(50), default="pending", index=True)
    params: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    logo_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("logo.id"),
        nullable=True,
        index=True,
    )
    product_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("products.id"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    product: Mapped[Product | None] = relationship(back_populates="jobs")
