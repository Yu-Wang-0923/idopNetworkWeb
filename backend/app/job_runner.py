"""Job runner: executes one run in a dedicated child process."""
from __future__ import annotations

import shutil
import sys
import traceback
from pathlib import Path, PurePosixPath

from .bootstrap import setup_runtime
from .config import get_settings
from .db import get_session_factory, init_db
from .jobs.context import ResolvedInput, RunCancelled, TaskContext
from .jobs.registry import get_task
from .models import Artifact, Dataset, Run, Workspace, utcnow
from .storage import Storage, sha256_file


def _resolve_inputs(db, run: Run, storage: Storage) -> list[ResolvedInput]:
    resolved: list[ResolvedInput] = []
    for ref in sorted(run.inputs, key=lambda item: (item.role, item.ordinal)):
        if ref.dataset_id:
            dataset = db.get(Dataset, ref.dataset_id)
            if dataset is None:
                raise FileNotFoundError(f"Dataset {ref.dataset_id} not found")
            resolved.append(
                ResolvedInput(
                    role=ref.role,
                    ordinal=ref.ordinal,
                    filename=dataset.original_filename,
                    path=storage.dataset_path(dataset.stored_path),
                    dataset_id=dataset.id,
                )
            )
        elif ref.artifact_id:
            artifact = db.get(Artifact, ref.artifact_id)
            if artifact is None:
                raise FileNotFoundError(f"Artifact {ref.artifact_id} not found")
            resolved.append(
                ResolvedInput(
                    role=ref.role,
                    ordinal=ref.ordinal,
                    filename=artifact.logical_name,
                    path=storage.artifact_path(artifact.run_id, artifact.relative_path),
                    artifact_id=artifact.id,
                )
            )
    return resolved


def _artifact_name(name: str) -> str:
    path = PurePosixPath(str(name).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not str(path):
        raise ValueError(f"Invalid artifact name: {name}")
    return str(path)


def _unique_name(name: str, used: set[str]) -> str:
    if name not in used:
        return name
    stem, dot, suffix = name.rpartition(".")
    counter = 2
    while True:
        candidate = f"{stem}-{counter}.{suffix}" if dot else f"{name}-{counter}"
        if candidate not in used:
            return candidate
        counter += 1


def _publish_artifacts(ctx: TaskContext, run: Run, db, storage: Storage) -> None:
    artifacts_dir = storage.run_artifacts_dir(run.id)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    published_bytes = 0
    for registered in ctx.registered_artifacts:
        relative = _unique_name(_artifact_name(registered.logical_name), used)
        used.add(relative)
        destination = storage.resolve_within(artifacts_dir, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(registered.source), str(destination))
        size = destination.stat().st_size
        published_bytes += size
        db.add(
            Artifact(
                run_id=run.id,
                workspace_id=run.workspace_id,
                logical_name=relative,
                relative_path=relative,
                semantic_type=registered.semantic_type,
                kind=registered.kind,
                mime=registered.mime,
                size_bytes=size,
                sha256=sha256_file(destination),
                is_primary=registered.is_primary,
            )
        )
    workspace = db.get(Workspace, run.workspace_id)
    if workspace is not None:
        workspace.storage_bytes = (workspace.storage_bytes or 0) + published_bytes
    db.commit()


def _finish(
    run_id: str,
    status: str,
    message_key: str,
    *,
    error_text: str | None = None,
    progress: int | None = None,
) -> None:
    with get_session_factory()() as db:
        run = db.get(Run, run_id)
        if run is None:
            return
        run.status = status
        run.message_key = message_key
        run.message_params_json = "{}"
        if error_text is not None:
            run.error_text = error_text[-4000:]
        if progress is not None:
            run.progress = progress
        run.finished_at = utcnow()
        run.heartbeat_at = utcnow()
        db.commit()


def execute_run(run_id: str) -> int:
    init_db()
    setup_runtime()
    settings = get_settings()
    storage = Storage(settings)
    session_factory = get_session_factory()

    with session_factory() as db:
        run = db.get(Run, run_id)
        if run is None:
            print(f"Run {run_id} not found", file=sys.stderr)
            return 2
        if run.status in {"succeeded", "failed", "cancelled"}:
            return 0
        if run.status == "queued":
            run.status = "running"
            run.started_at = run.started_at or utcnow()
        spec = get_task(run.task_key)
        if spec is None:
            _finish(run_id, "failed", "run.message.failed", error_text="Unknown task")
            return 1
        try:
            params = spec.params_model.model_validate_json(run.params_json).model_dump()
        except Exception as exc:
            _finish(
                run_id,
                "failed",
                "run.message.failed",
                error_text=f"Invalid parameters: {exc}",
            )
            return 1
        inputs = _resolve_inputs(db, run, storage)
        run.heartbeat_at = utcnow()
        db.commit()

    work_dir = storage.run_work_dir(run_id)
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    ctx = TaskContext(
        run_id=run_id,
        workspace_id=run.workspace_id,
        task_key=run.task_key,
        params=params,
        inputs=inputs,
        work_dir=work_dir,
        session_factory=session_factory,
        storage=storage,
        settings=settings,
    )

    try:
        ctx.log(raw_text=f"Task {run.task_key} started")
        spec.run(ctx)
        ctx.check_cancelled()
        with session_factory() as db:
            current = db.get(Run, run_id)
            if current is None:
                return 2
            _publish_artifacts(ctx, current, db, storage)
        _finish(run_id, "succeeded", "run.message.succeeded", progress=100)
        shutil.rmtree(work_dir, ignore_errors=True)
        return 0
    except RunCancelled:
        ctx.log(raw_text="Run cancelled by request")
        _finish(run_id, "cancelled", "run.message.cancelled")
        return 0
    except Exception:
        detail = traceback.format_exc()
        ctx.log(level="error", raw_text=detail)
        _finish(run_id, "failed", "run.message.failed", error_text=detail)
        return 1


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print("usage: python -m app.job_runner <run_id>", file=sys.stderr)
        return 2
    return execute_run(argv[0])


if __name__ == "__main__":
    raise SystemExit(main())
