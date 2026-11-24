"""Dataset upload and inspection endpoints."""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..db import get_db
from ..file_inspect import inspect_upload
from ..models import Dataset, RunInput, Workspace
from ..schemas import DatasetOut, InspectOut
from ..serializers import dataset_out
from ..storage import Storage, StorageError, UploadTooLarge, detect_kind, safe_filename
from .deps import get_workspace, rate_limit, verify_origin

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _get_owned_dataset(db: Session, dataset_id: str, workspace: Workspace) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None or dataset.workspace_id != workspace.id:
        raise HTTPException(status_code=404, detail={"code": "dataset.not_found"})
    return dataset


@router.post("", response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
def upload_dataset(
    file: UploadFile = File(...),
    workspace: Workspace = Depends(get_workspace),
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
    _origin: None = Depends(verify_origin),
    _rate: None = Depends(rate_limit),
) -> DatasetOut:
    filename = safe_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".zip"}:
        raise HTTPException(status_code=400, detail={"code": "upload.unsupported_type"})
    max_bytes = settings.max_zip_bytes if suffix == ".zip" else settings.max_csv_bytes

    storage = Storage(settings)
    dataset = Dataset(
        workspace_id=workspace.id,
        original_filename=filename,
        stored_path="",
        size_bytes=0,
        sha256="",
        detected_kind="unknown",
        content_type=file.content_type or "application/octet-stream",
    )
    db.add(dataset)
    db.flush()
    try:
        stored_path, size, digest, stored_name, mime = storage.save_upload(
            dataset.id, filename, file.file, max_bytes
        )
    except UploadTooLarge as exc:
        db.rollback()
        raise HTTPException(status_code=413, detail={"code": "upload.too_large"}) from exc
    except Exception:
        db.rollback()
        raise

    path = storage.dataset_path(stored_path)
    try:
        kind, _names = detect_kind(path, settings)
    except StorageError as exc:
        shutil.rmtree(storage.dataset_dir(dataset.id), ignore_errors=True)
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail={"code": "upload.invalid_archive", "message": str(exc)},
        ) from exc

    dataset.original_filename = filename
    dataset.stored_path = stored_path
    dataset.size_bytes = size
    dataset.sha256 = digest
    dataset.detected_kind = kind
    dataset.content_type = mime
    workspace.storage_bytes = (workspace.storage_bytes or 0) + size
    db.commit()
    db.refresh(dataset)
    return dataset_out(dataset)


@router.get("", response_model=list[DatasetOut])
def list_datasets(
    workspace: Workspace = Depends(get_workspace),
    db: Session = Depends(get_db),
) -> list[DatasetOut]:
    datasets = db.scalars(
        select(Dataset)
        .where(Dataset.workspace_id == workspace.id)
        .order_by(Dataset.created_at.desc())
    ).all()
    return [dataset_out(item) for item in datasets]


@router.get("/{dataset_id}/inspect", response_model=InspectOut)
def inspect_dataset(
    dataset_id: str,
    workspace: Workspace = Depends(get_workspace),
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
) -> dict:
    dataset = _get_owned_dataset(db, dataset_id, workspace)
    storage = Storage(settings)
    try:
        result = inspect_upload(
            storage.dataset_path(dataset.stored_path),
            dataset.original_filename,
            settings,
        )
    except StorageError as exc:
        raise HTTPException(
            status_code=400, detail={"code": "upload.invalid_archive", "message": str(exc)}
        ) from exc
    return result


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_id: str,
    workspace: Workspace = Depends(get_workspace),
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
    _origin: None = Depends(verify_origin),
) -> None:
    dataset = _get_owned_dataset(db, dataset_id, workspace)
    referenced = db.scalar(
        select(RunInput.id).where(RunInput.dataset_id == dataset.id).limit(1)
    )
    if referenced:
        raise HTTPException(status_code=409, detail={"code": "dataset.in_use"})
    storage = Storage(settings)
    shutil.rmtree(storage.dataset_dir(dataset.id), ignore_errors=True)
    workspace.storage_bytes = max(
        0, (workspace.storage_bytes or 0) - (dataset.size_bytes or 0)
    )
    db.delete(dataset)
    db.commit()
