"""Execution context passed to computational tasks."""
from __future__ import annotations

import json
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ..config import Settings
from ..models import JobLog, Run, utcnow
from ..storage import Storage, safe_filename


class RunCancelled(Exception):
    """Raised when a task observes a cancellation request."""


@dataclass
class ResolvedInput:
    role: str
    ordinal: int
    filename: str
    path: Path
    dataset_id: str | None = None
    artifact_id: str | None = None


@dataclass
class RegisteredArtifact:
    source: Path
    logical_name: str
    semantic_type: str
    kind: str
    mime: str
    is_primary: bool


@dataclass
class TaskContext:
    run_id: str
    workspace_id: str
    task_key: str
    params: dict
    inputs: list[ResolvedInput]
    work_dir: Path
    session_factory: sessionmaker[Session]
    storage: Storage
    settings: Settings
    _progress: int = 0
    _seq: int | None = None
    _artifacts: list[RegisteredArtifact] = field(default_factory=list)

    @property
    def output_dir(self) -> Path:
        path = self.work_dir / "output"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _next_seq(self, db: Session) -> int:
        if self._seq is None:
            current = db.scalar(
                select(func.max(JobLog.seq)).where(JobLog.run_id == self.run_id)
            )
            self._seq = int(current or 0)
        self._seq += 1
        return self._seq

    def _write_log(
        self,
        db: Session,
        level: str,
        message_key: str | None,
        message_params: dict,
        raw_text: str | None,
    ) -> None:
        db.add(
            JobLog(
                run_id=self.run_id,
                seq=self._next_seq(db),
                ts=utcnow(),
                level=level,
                message_key=message_key,
                message_params_json=json.dumps(message_params, ensure_ascii=False),
                raw_text=raw_text,
            )
        )

    def progress(
        self,
        stage_key: str,
        fraction: float,
        message_key: str | None = None,
        **message_params,
    ) -> None:
        fraction = max(0.0, min(1.0, float(fraction)))
        self._progress = max(self._progress, int(round(fraction * 100)))
        with self.session_factory() as db:
            run = db.get(Run, self.run_id)
            if run is None:
                return
            run.progress = self._progress
            run.stage_key = stage_key
            if message_key:
                run.message_key = message_key
                run.message_params_json = json.dumps(message_params, ensure_ascii=False)
            run.heartbeat_at = utcnow()
            if message_key:
                self._write_log(db, "info", message_key, message_params, None)
            db.commit()
        if message_key:
            print(f"[progress {self._progress:3d}%] {message_key} {message_params}", flush=True)

    def log(
        self,
        message_key: str | None = None,
        *,
        level: str = "info",
        raw_text: str | None = None,
        **message_params,
    ) -> None:
        with self.session_factory() as db:
            self._write_log(db, level, message_key, message_params, raw_text)
            db.commit()
        rendered = raw_text if raw_text is not None else f"{message_key} {message_params}"
        print(f"[{level}] {rendered}", flush=True)

    def check_cancelled(self) -> None:
        with self.session_factory() as db:
            requested = db.scalar(
                select(Run.cancel_requested).where(Run.id == self.run_id)
            )
        if requested:
            raise RunCancelled()

    def add_artifact(
        self,
        path: str | Path,
        logical_name: str,
        semantic_type: str,
        *,
        kind: str | None = None,
        mime: str | None = None,
        is_primary: bool = False,
    ) -> None:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Artifact file does not exist: {source}")
        source_resolved = source.resolve()
        if not source_resolved.is_relative_to(self.output_dir.resolve()):
            raise ValueError("Artifacts must be written inside the task output directory")
        relative = source_resolved.relative_to(self.output_dir.resolve())
        name = logical_name.strip().replace("\\", "/")
        if not name or name.startswith("/") or ".." in Path(name).parts:
            name = safe_filename(source.name)
        guessed = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        self._artifacts.append(
            RegisteredArtifact(
                source=source_resolved,
                logical_name=name,
                semantic_type=semantic_type,
                kind=kind or source.suffix.lstrip(".").lower() or "bin",
                mime=mime or guessed,
                is_primary=is_primary,
            )
        )

    @property
    def registered_artifacts(self) -> list[RegisteredArtifact]:
        return list(self._artifacts)


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
