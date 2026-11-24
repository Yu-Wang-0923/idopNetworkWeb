"""Database models."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """Store naive UTC in SQLite, return timezone-aware datetimes."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    run_count: Mapped[int] = mapped_column(Integer, default=0)
    storage_bytes: Mapped[int] = mapped_column(Integer, default=0)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    original_filename: Mapped[str] = mapped_column(String(512))
    stored_path: Mapped[str] = mapped_column(String(1024))
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    detected_kind: Mapped[str] = mapped_column(String(32), default="unknown")
    content_type: Mapped[str] = mapped_column(String(128), default="application/octet-stream")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    __table_args__ = (
        Index("ix_datasets_workspace_created", "workspace_id", "created_at"),
    )


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    module: Mapped[str] = mapped_column(String(32))
    task_key: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="queued")
    params_json: Mapped[str] = mapped_column(Text, default="{}")
    params_version: Mapped[int] = mapped_column(Integer, default=1)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    stage_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    message_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    message_params_json: Mapped[str] = mapped_column(Text, default="{}")
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    inputs: Mapped[list[RunInput]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    artifacts: Mapped[list[Artifact]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    logs: Mapped[list[JobLog]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_runs_status_created", "status", "created_at"),
        Index("ix_runs_workspace_created", "workspace_id", "created_at"),
    )


class RunInput(Base):
    __tablename__ = "run_inputs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(32))
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    dataset_id: Mapped[str | None] = mapped_column(
        ForeignKey("datasets.id", ondelete="RESTRICT"), nullable=True
    )
    artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("artifacts.id", ondelete="RESTRICT"), nullable=True
    )

    run: Mapped[Run] = relationship(back_populates="inputs")

    __table_args__ = (
        UniqueConstraint("run_id", "role", "ordinal", name="uq_run_input_slot"),
    )


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    logical_name: Mapped[str] = mapped_column(String(512))
    relative_path: Mapped[str] = mapped_column(String(1024))
    semantic_type: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(16))
    mime: Mapped[str] = mapped_column(String(128))
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    run: Mapped[Run] = relationship(back_populates="artifacts")


class JobLog(Base):
    __tablename__ = "job_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer)
    ts: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    level: Mapped[str] = mapped_column(String(16), default="info")
    message_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    message_params_json: Mapped[str] = mapped_column(Text, default="{}")
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped[Run] = relationship(back_populates="logs")

    __table_args__ = (
        UniqueConstraint("run_id", "seq", name="uq_job_log_seq"),
    )
