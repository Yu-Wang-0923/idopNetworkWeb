"""Artifact metadata, downloads, inline content, and table previews."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..db import get_db
from ..file_inspect import curve_feature_count
from ..models import Artifact, Workspace
from ..schemas import ArtifactOut
from ..serializers import artifact_out
from ..storage import Storage, StorageError, inspect_zip, zip_conditions, zip_members
from .deps import get_workspace

router = APIRouter(prefix="/artifacts", tags=["artifacts"])

_HTML_CSP = (
    "sandbox allow-scripts allow-downloads; default-src 'none'; "
    "style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data: blob:"
)


def _get_artifact(db: Session, artifact_id: str) -> Artifact:
    artifact = db.get(Artifact, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail={"code": "artifact.not_found"})
    return artifact


def _artifact_path(artifact: Artifact, settings: Settings) -> Path:
    storage = Storage(settings)
    try:
        path = storage.artifact_path(artifact.run_id, artifact.relative_path)
    except StorageError as exc:
        raise HTTPException(status_code=400, detail={"code": "artifact.invalid_path"}) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail={"code": "artifact.file_missing"})
    return path


@router.get("", response_model=list[ArtifactOut])
def list_artifacts(
    workspace: Workspace = Depends(get_workspace),
    db: Session = Depends(get_db),
    semantic_type: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=200),
) -> list[ArtifactOut]:
    statement = select(Artifact).where(Artifact.workspace_id == workspace.id)
    if semantic_type:
        statement = statement.where(Artifact.semantic_type == semantic_type)
    statement = statement.order_by(Artifact.created_at.desc()).limit(limit)
    return [artifact_out(item) for item in db.scalars(statement).all()]


@router.get("/{artifact_id}", response_model=ArtifactOut)
def read_artifact(artifact_id: str, db: Session = Depends(get_db)) -> ArtifactOut:
    return artifact_out(_get_artifact(db, artifact_id))


@router.get("/{artifact_id}/download")
def download_artifact(
    artifact_id: str,
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
) -> FileResponse:
    artifact = _get_artifact(db, artifact_id)
    path = _artifact_path(artifact, settings)
    filename = Path(artifact.logical_name).name or artifact.id
    return FileResponse(path, media_type=artifact.mime, filename=filename)


@router.get("/{artifact_id}/content")
def inline_artifact(
    artifact_id: str,
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
) -> FileResponse:
    artifact = _get_artifact(db, artifact_id)
    path = _artifact_path(artifact, settings)
    headers = {
        "Content-Disposition": "inline",
        "X-Content-Type-Options": "nosniff",
    }
    if artifact.kind == "html":
        headers["Content-Security-Policy"] = _HTML_CSP
    return FileResponse(path, media_type=artifact.mime, headers=headers)


@router.get("/{artifact_id}/table")
def artifact_table(
    artifact_id: str,
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> dict:
    artifact = _get_artifact(db, artifact_id)
    if artifact.kind != "csv":
        raise HTTPException(status_code=400, detail={"code": "artifact.not_a_table"})
    path = _artifact_path(artifact, settings)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"code": "artifact.unreadable_table"}) from exc
    slice_ = frame.iloc[offset : offset + limit]
    rows = []
    for record in slice_.itertuples(index=False, name=None):
        rows.append([None if (isinstance(value, float) and math.isnan(value)) else value for value in record])
    return {
        "columns": [str(column) for column in frame.columns],
        "rows": rows,
        "total": int(len(frame)),
    }


@router.get("/{artifact_id}/inspect")
def inspect_artifact(
    artifact_id: str,
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
) -> dict:
    artifact = _get_artifact(db, artifact_id)
    if artifact.kind != "zip":
        raise HTTPException(status_code=400, detail={"code": "artifact.not_an_archive"})
    path = _artifact_path(artifact, settings)
    try:
        names = inspect_zip(path, settings)
    except StorageError as exc:
        raise HTTPException(status_code=400, detail={"code": "artifact.invalid_archive"}) from exc
    return {
        "kind": artifact.semantic_type,
        "conditions": zip_conditions(names),
        "members": zip_members(names),
        "features": curve_feature_count(path)
        if artifact.semantic_type == "curve_fitting_export"
        else None,
    }
