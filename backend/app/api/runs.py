"""Run creation, listing, cancellation, and logs."""
from __future__ import annotations

import json
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..db import get_db
from ..jobs.registry import get_task
from ..models import Artifact, Dataset, JobLog, Run, RunInput, Workspace, utcnow
from ..schemas import ArtifactOut, InputRef, LogOut, RunCreate, RunOut, RunSummary
from ..serializers import run_out, run_summary
from .deps import get_run_read, get_run_write, get_workspace, rate_limit, verify_origin

router = APIRouter(prefix="/runs", tags=["runs"])


def _check_quota(db: Session, workspace: Workspace, settings: Settings) -> None:
    concurrent = db.scalar(
        select(func.count())
        .select_from(Run)
        .where(
            Run.workspace_id == workspace.id,
            Run.status.in_(("queued", "running")),
        )
    )
    if (concurrent or 0) >= settings.max_concurrent_runs_per_workspace:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"code": "run.too_many_active"},
        )
    since = utcnow() - timedelta(days=1)
    daily = db.scalar(
        select(func.count())
        .select_from(Run)
        .where(Run.workspace_id == workspace.id, Run.created_at >= since)
    )
    if (daily or 0) >= settings.max_runs_per_day_per_workspace:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"code": "run.daily_limit"},
        )


def _resolve_inputs(
    db: Session, workspace: Workspace, inputs: list[InputRef], allowed_roles: set[str]
) -> list[InputRef]:
    slots: set[tuple[str, int]] = set()
    for ref in inputs:
        if ref.role not in allowed_roles:
            raise HTTPException(
                status_code=400,
                detail={"code": "run.invalid_input_role", "role": ref.role},
            )
        slot = (ref.role, ref.ordinal)
        if slot in slots:
            raise HTTPException(status_code=400, detail={"code": "run.duplicate_input"})
        slots.add(slot)
        if ref.dataset_id:
            dataset = db.get(Dataset, ref.dataset_id)
            if dataset is None or dataset.workspace_id != workspace.id:
                raise HTTPException(
                    status_code=404, detail={"code": "dataset.not_found"}
                )
        else:
            artifact = db.get(Artifact, ref.artifact_id)
            if artifact is None:
                raise HTTPException(status_code=404, detail={"code": "artifact.not_found"})
            if artifact.workspace_id != workspace.id:
                raise HTTPException(
                    status_code=403, detail={"code": "artifact.not_owned"}
                )
    return inputs


@router.post("", response_model=RunOut, status_code=status.HTTP_202_ACCEPTED)
def create_run(
    payload: RunCreate,
    workspace: Workspace = Depends(get_workspace),
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
    _origin: None = Depends(verify_origin),
    _rate: None = Depends(rate_limit),
) -> RunOut:
    spec = get_task(payload.task_key)
    if spec is None or not spec.enabled:
        raise HTTPException(status_code=404, detail={"code": "task.not_found"})
    try:
        validated = spec.params_model.model_validate(payload.params)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "run.invalid_params",
                "errors": json.loads(exc.json(include_url=False)),
            },
        ) from exc

    _check_quota(db, workspace, settings)
    _resolve_inputs(db, workspace, payload.inputs, set(spec.input_roles))

    run = Run(
        workspace_id=workspace.id,
        module=spec.module,
        task_key=spec.key,
        status="queued",
        params_json=validated.model_dump_json(),
        params_version=1,
        progress=0,
        message_key="run.message.queued",
    )
    db.add(run)
    db.flush()
    for ref in payload.inputs:
        db.add(
            RunInput(
                run_id=run.id,
                role=ref.role,
                ordinal=ref.ordinal,
                dataset_id=ref.dataset_id,
                artifact_id=ref.artifact_id,
            )
        )
    workspace.run_count = (workspace.run_count or 0) + 1
    db.commit()
    db.refresh(run)
    return run_out(db, run)


@router.get("", response_model=list[RunSummary])
def list_runs(
    workspace: Workspace = Depends(get_workspace),
    db: Session = Depends(get_db),
    module: str | None = Query(default=None),
    run_status: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[RunSummary]:
    statement = select(Run).where(Run.workspace_id == workspace.id)
    if module:
        statement = statement.where(Run.module == module)
    if run_status:
        statement = statement.where(Run.status == run_status)
    statement = statement.order_by(Run.created_at.desc()).limit(limit).offset(offset)
    runs = db.scalars(statement).all()
    return [run_summary(run) for run in runs]


@router.get("/{run_id}", response_model=RunOut)
def read_run(
    run_id: str,
    db: Session = Depends(get_db),
) -> RunOut:
    run = get_run_read(db, run_id)
    return run_out(db, run)


@router.post("/{run_id}/cancel", response_model=RunOut)
def cancel_run(
    run_id: str,
    workspace: Workspace = Depends(get_workspace),
    db: Session = Depends(get_db),
    _origin: None = Depends(verify_origin),
    _rate: None = Depends(rate_limit),
) -> RunOut:
    run = get_run_write(db, run_id, workspace)
    if run.status in {"succeeded", "failed", "cancelled"}:
        raise HTTPException(status_code=409, detail={"code": "run.finished"})
    if run.status == "queued":
        run.status = "cancelled"
        run.message_key = "run.message.cancelled"
        run.finished_at = utcnow()
    else:
        run.cancel_requested = True
    db.commit()
    db.refresh(run)
    return run_out(db, run)


@router.post("/{run_id}/retry", response_model=RunOut, status_code=status.HTTP_202_ACCEPTED)
def retry_run(
    run_id: str,
    workspace: Workspace = Depends(get_workspace),
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
    _origin: None = Depends(verify_origin),
    _rate: None = Depends(rate_limit),
) -> RunOut:
    original = get_run_write(db, run_id, workspace)
    _check_quota(db, workspace, settings)
    retry = Run(
        workspace_id=workspace.id,
        module=original.module,
        task_key=original.task_key,
        status="queued",
        params_json=original.params_json,
        params_version=original.params_version,
        progress=0,
        message_key="run.message.queued",
    )
    db.add(retry)
    db.flush()
    for ref in original.inputs:
        db.add(
            RunInput(
                run_id=retry.id,
                role=ref.role,
                ordinal=ref.ordinal,
                dataset_id=ref.dataset_id,
                artifact_id=ref.artifact_id,
            )
        )
    workspace.run_count = (workspace.run_count or 0) + 1
    db.commit()
    db.refresh(retry)
    return run_out(db, retry)


@router.get("/{run_id}/logs", response_model=list[LogOut])
def read_logs(
    run_id: str,
    db: Session = Depends(get_db),
    after_seq: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=2000),
) -> list[dict]:
    get_run_read(db, run_id)
    logs = db.scalars(
        select(JobLog)
        .where(JobLog.run_id == run_id, JobLog.seq > after_seq)
        .order_by(JobLog.seq)
        .limit(limit)
    ).all()
    result = []
    for log in logs:
        try:
            params = json.loads(log.message_params_json or "{}")
        except ValueError:
            params = {}
        result.append(
            {
                "seq": log.seq,
                "ts": log.ts,
                "level": log.level,
                "message_key": log.message_key,
                "message_params": params,
                "raw_text": log.raw_text,
            }
        )
    return result
