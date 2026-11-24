"""Shared API dependencies."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..db import get_db
from ..models import Run, Workspace, utcnow

_lock = threading.Lock()
_hits: dict[str, deque[float]] = defaultdict(deque)


def get_workspace(
    request: Request,
    db: Session = Depends(get_db),
) -> Workspace:
    workspace_id = getattr(request.state, "workspace_id", None)
    if not workspace_id:
        raise HTTPException(status_code=500, detail={"code": "session.missing"})
    workspace = db.get(Workspace, workspace_id)
    if workspace is None:
        workspace = Workspace(id=workspace_id)
        db.add(workspace)
        db.commit()
        db.refresh(workspace)
    else:
        workspace.last_seen_at = utcnow()
        db.commit()
    return workspace


def verify_origin(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    origin = request.headers.get("origin")
    if origin and origin not in settings.allowed_origins_list:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "request.forbidden_origin"},
        )


def rate_limit(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    client = request.client.host if request.client else "unknown"
    key = f"{client}:{request.url.path}"
    now = time.monotonic()
    window_start = now - 60.0
    with _lock:
        bucket = _hits[key]
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        if len(bucket) >= settings.rate_limit_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"code": "rate.limited"},
            )
        bucket.append(now)


def get_run_read(db: Session, run_id: str) -> Run:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail={"code": "run.not_found"})
    return run


def get_run_write(db: Session, run_id: str, workspace: Workspace) -> Run:
    run = get_run_read(db, run_id)
    if run.workspace_id != workspace.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "run.not_owned"},
        )
    return run
