"""Worker process: claims queued runs and supervises one child process per run."""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from .bootstrap import setup_runtime
from .config import get_settings
from .db import get_session_factory, init_db
from .jobs.queue import claim_next_run, fail_run, mark_stale_runs, set_heartbeat
from .jobs.registry import get_task
from .models import Run, utcnow
from .storage import Storage

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _log_tail(path: Path, limit: int = 4000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def execute_run(run_id: str, worker_id: str) -> None:
    settings = get_settings()
    storage = Storage(settings)
    session_factory = get_session_factory()

    with session_factory() as db:
        run = db.get(Run, run_id)
        if run is None:
            return
        task_key = run.task_key
    spec = get_task(task_key)
    timeout_seconds = spec.timeout_s if spec else 3600

    log_path = storage.run_dir(run_id) / "job.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = str(BACKEND_ROOT)
    environment["IDOP_DATA_DIR"] = str(settings.data_dir)
    environment["OMP_NUM_THREADS"] = "1"

    timed_out = False
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, "-m", "app.job_runner", run_id],
            cwd=str(BACKEND_ROOT),
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        started = time.monotonic()
        last_heartbeat = 0.0
        cancel_deadline: float | None = None

        while process.poll() is None:
            now = time.monotonic()
            if now - last_heartbeat >= settings.worker_heartbeat_seconds:
                set_heartbeat(run_id, worker_id)
                last_heartbeat = now

            with session_factory() as db:
                current = db.get(Run, run_id)
                cancel_requested = bool(current and current.cancel_requested)

            if cancel_requested:
                if cancel_deadline is None:
                    cancel_deadline = now + settings.cancel_grace_seconds
                elif now >= cancel_deadline:
                    _terminate(process)
                    break

            if now - started > timeout_seconds:
                timed_out = True
                _terminate(process)
                break

            time.sleep(0.4)

    if timed_out:
        fail_run(run_id, f"Task exceeded timeout of {timeout_seconds} seconds")
        return

    with session_factory() as db:
        run = db.get(Run, run_id)
        if run is None or run.status != "running":
            return
        if run.cancel_requested:
            run.status = "cancelled"
            run.message_key = "run.message.cancelled"
        else:
            tail = _log_tail(log_path)
            run.status = "failed"
            run.message_key = "run.message.failed"
            run.error_text = tail or "Child process exited without recording a result"
        run.finished_at = utcnow()
        db.commit()


def worker_loop(once: bool = False) -> None:
    settings = get_settings()
    init_db()
    setup_runtime()
    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    print(f"[worker] {worker_id} started", flush=True)
    while True:
        mark_stale_runs(settings.worker_stale_seconds)
        run_id = claim_next_run(worker_id)
        if run_id:
            print(f"[worker] claimed {run_id}", flush=True)
            execute_run(run_id, worker_id)
            if once:
                return
        else:
            if once:
                return
            time.sleep(settings.worker_poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="idopNetwork Web worker")
    parser.add_argument("--once", action="store_true", help="Process at most one run and exit")
    args = parser.parse_args(argv)
    try:
        worker_loop(once=args.once)
    except KeyboardInterrupt:
        print("[worker] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
