"""SQLite-backed job queue operations."""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import func, select, update

from ..db import get_engine, get_session_factory
from ..models import Run, utcnow


def claim_next_run(worker_id: str) -> str | None:
    """Atomically move the oldest queued run to running and return its id."""
    now = utcnow()
    next_id = (
        select(Run.id)
        .where(Run.status == "queued")
        .order_by(Run.created_at)
        .limit(1)
        .scalar_subquery()
    )
    statement = (
        update(Run)
        .where(Run.id == next_id)
        .values(
            status="running",
            worker_id=worker_id,
            started_at=func.coalesce(Run.started_at, now),
            heartbeat_at=now,
            attempt=Run.attempt + 1,
        )
        .returning(Run.id)
    )
    with get_engine().begin() as connection:
        row = connection.execute(statement).first()
    return row[0] if row else None


def set_heartbeat(run_id: str, worker_id: str) -> None:
    with get_session_factory()() as db:
        db.execute(
            update(Run)
            .where(Run.id == run_id, Run.worker_id == worker_id, Run.status == "running")
            .values(heartbeat_at=utcnow())
        )
        db.commit()


def mark_stale_runs(stale_seconds: float) -> int:
    cutoff = utcnow() - timedelta(seconds=stale_seconds)
    with get_session_factory()() as db:
        result = db.execute(
            update(Run)
            .where(
                Run.status == "running",
                Run.heartbeat_at.is_not(None),
                Run.heartbeat_at < cutoff,
            )
            .values(
                status="failed",
                error_text="worker heartbeat lost",
                message_key="run.message.failed",
                finished_at=utcnow(),
            )
        )
        db.commit()
        return int(result.rowcount or 0)


def fail_run(run_id: str, error_text: str) -> None:
    with get_session_factory()() as db:
        run = db.get(Run, run_id)
        if run is None or run.status not in {"queued", "running"}:
            return
        run.status = "failed"
        run.error_text = error_text[-4000:]
        run.message_key = "run.message.failed"
        run.finished_at = utcnow()
        db.commit()
