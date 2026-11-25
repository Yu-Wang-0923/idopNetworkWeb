from __future__ import annotations

import uuid
from datetime import timedelta

from sqlalchemy import select

from app.cli import run_cleanup
from app.config import get_settings
from app.db import get_session_factory
from app.models import Artifact, Dataset, Run, RunInput, Workspace, utcnow
from app.storage import Storage, sha256_file


def _make_workspace() -> str:
    with get_session_factory()() as db:
        workspace = Workspace()
        db.add(workspace)
        db.commit()
        db.refresh(workspace)
        return workspace.id


def _make_run(workspace_id: str, *, status: str, age_days: int, with_artifact: bool) -> str:
    settings = get_settings()
    storage = Storage(settings)
    run_id = str(uuid.uuid4())
    run_dir = storage.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "job.log").write_text("log", encoding="utf-8")
    work_dir = storage.run_work_dir(run_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "tmp.bin").write_bytes(b"x" * 10)
    created = utcnow() - timedelta(days=age_days)
    with get_session_factory()() as db:
        db.add(
            Run(
                id=run_id,
                workspace_id=workspace_id,
                module="netrecon",
                task_key="netrecon.single",
                status=status,
                created_at=created,
                finished_at=created if status != "running" else None,
            )
        )
        if with_artifact:
            artifact_dir = storage.run_artifacts_dir(run_id)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            path = artifact_dir / "network.html"
            path.write_bytes(b"<html>network</html>")
            db.add(
                Artifact(
                    run_id=run_id,
                    workspace_id=workspace_id,
                    logical_name="network.html",
                    relative_path="network.html",
                    semantic_type="netrecon_network_html",
                    kind="html",
                    mime="text/html",
                    size_bytes=path.stat().st_size,
                    sha256=sha256_file(path),
                )
            )
        db.commit()
    return run_id


def _make_dataset(workspace_id: str, *, age_days: int, size: int) -> str:
    settings = get_settings()
    storage = Storage(settings)
    dataset_id = str(uuid.uuid4())
    directory = storage.dataset_dir(dataset_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "data.csv"
    path.write_bytes(b"x" * size)
    with get_session_factory()() as db:
        db.add(
            Dataset(
                id=dataset_id,
                workspace_id=workspace_id,
                original_filename="data.csv",
                stored_path=str(path.relative_to(settings.data_dir)).replace("\\", "/"),
                size_bytes=size,
                sha256=sha256_file(path),
                created_at=utcnow() - timedelta(days=age_days),
            )
        )
        db.commit()
    return dataset_id


def test_cleanup_removes_old_runs_and_unreferenced_uploads():
    settings = get_settings()
    storage = Storage(settings)
    workspace_id = _make_workspace()

    old_run = _make_run(workspace_id, status="succeeded", age_days=60, with_artifact=True)
    new_run = _make_run(workspace_id, status="succeeded", age_days=1, with_artifact=True)
    active_run = _make_run(workspace_id, status="running", age_days=60, with_artifact=False)

    old_free = _make_dataset(workspace_id, age_days=60, size=20)
    old_used = _make_dataset(workspace_id, age_days=60, size=30)
    new_free = _make_dataset(workspace_id, age_days=1, size=40)
    with get_session_factory()() as db:
        db.add(RunInput(run_id=new_run, role="csv", ordinal=0, dataset_id=old_used))
        db.commit()

    dry = run_cleanup(60 - 1, dry_run=True)
    assert dry["runs_deleted"] == 1
    assert storage.run_dir(old_run).exists()
    with get_session_factory()() as db:
        assert db.get(Run, old_run) is not None

    summary = run_cleanup(60 - 1)
    assert summary["runs_deleted"] == 1
    assert summary["active_runs_skipped"] == 1
    assert summary["datasets_deleted"] == 1
    assert summary["bytes_freed"] > 0
    assert not storage.run_dir(old_run).exists()
    assert not storage.run_work_dir(old_run).exists()
    assert storage.run_dir(new_run).exists()
    assert storage.run_dir(active_run).exists()
    assert not storage.dataset_dir(old_free).exists()
    assert storage.dataset_dir(old_used).exists()
    assert storage.dataset_dir(new_free).exists()

    with get_session_factory()() as db:
        assert db.get(Run, old_run) is None
        assert db.get(Run, new_run) is not None
        assert db.get(Run, active_run) is not None
        assert db.get(Dataset, old_free) is None
        assert db.get(Dataset, old_used) is not None
        assert db.get(Dataset, new_free) is not None
        remaining_artifacts = db.scalars(
            select(Artifact).where(Artifact.run_id == old_run)
        ).all()
        assert remaining_artifacts == []
        workspace = db.get(Workspace, workspace_id)
        assert workspace.storage_bytes == 30 + 40 + 20
