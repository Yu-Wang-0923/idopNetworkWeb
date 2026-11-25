"""Maintenance CLI: python -m app.cli cleanup --days 30 [--dry-run]."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import timedelta
from pathlib import Path

from sqlalchemy import delete, func, select

from .config import get_settings
from .db import get_session_factory, init_db
from .models import Artifact, Dataset, Run, RunInput, Workspace, utcnow
from .storage import Storage


def _dir_size(path: Path) -> int:
    try:
        return sum(
            item.stat().st_size for item in path.rglob("*") if item.is_file()
        )
    except OSError:
        return 0


def run_cleanup(days: int, *, dry_run: bool = False) -> dict:
    if days < 0:
        raise ValueError("days must be non-negative")
    settings = get_settings()
    storage = Storage(settings)
    cutoff = utcnow() - timedelta(days=days)
    summary: dict[str, object] = {
        "cutoff": cutoff.isoformat(),
        "days": days,
        "dry_run": dry_run,
        "runs_deleted": 0,
        "datasets_deleted": 0,
        "work_dirs_removed": 0,
        "active_runs_skipped": 0,
        "bytes_freed": 0,
    }
    with get_session_factory()() as db:
        old_runs = db.scalars(select(Run).where(Run.created_at < cutoff)).all()
        deletable = [
            run for run in old_runs if run.status not in {"queued", "running"}
        ]
        summary["active_runs_skipped"] = len(old_runs) - len(deletable)
        for run in deletable:
            for directory in (
                storage.run_dir(run.id),
                storage.run_work_dir(run.id),
            ):
                if directory.exists():
                    summary["bytes_freed"] += _dir_size(directory)
                    if not dry_run:
                        shutil.rmtree(directory, ignore_errors=True)
            if not dry_run:
                db.execute(delete(Run).where(Run.id == run.id))
        summary["runs_deleted"] = len(deletable)
        if not dry_run:
            db.flush()
            remaining_ids = {row[0] for row in db.execute(select(Run.id)).all()}
            if settings.work_dir.exists():
                for directory in settings.work_dir.iterdir():
                    if not directory.is_dir() or directory.name in remaining_ids:
                        continue
                    summary["work_dirs_removed"] += 1
                    summary["bytes_freed"] += _dir_size(directory)
                    shutil.rmtree(directory, ignore_errors=True)

        referenced = select(RunInput.dataset_id).where(
            RunInput.dataset_id.is_not(None)
        )
        old_datasets = db.scalars(
            select(Dataset).where(
                Dataset.created_at < cutoff, ~Dataset.id.in_(referenced)
            )
        ).all()
        for dataset in old_datasets:
            directory = storage.dataset_dir(dataset.id)
            if directory.exists():
                summary["bytes_freed"] += _dir_size(directory)
                if not dry_run:
                    shutil.rmtree(directory, ignore_errors=True)
            if not dry_run:
                db.execute(delete(Dataset).where(Dataset.id == dataset.id))
        summary["datasets_deleted"] = len(old_datasets)

        if not dry_run:
            db.flush()
            for workspace in db.scalars(select(Workspace)).all():
                uploaded = db.scalar(
                    select(func.coalesce(func.sum(Dataset.size_bytes), 0)).where(
                        Dataset.workspace_id == workspace.id
                    )
                )
                produced = db.scalar(
                    select(func.coalesce(func.sum(Artifact.size_bytes), 0)).where(
                        Artifact.workspace_id == workspace.id
                    )
                )
                workspace.storage_bytes = int(uploaded or 0) + int(produced or 0)
            db.commit()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli", description="idopNetwork Web maintenance commands"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    cleanup = commands.add_parser(
        "cleanup", help="Delete runs and uploads older than the retention window"
    )
    cleanup.add_argument(
        "--days",
        type=int,
        default=None,
        help="Retention window in days (default: IDOP_RETENTION_DAYS)",
    )
    cleanup.add_argument(
        "--dry-run", action="store_true", help="Report what would be deleted"
    )
    args = parser.parse_args(argv)

    if args.command == "cleanup":
        init_db()
        days = args.days if args.days is not None else get_settings().retention_days
        print(json.dumps(run_cleanup(days, dry_run=args.dry_run), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
