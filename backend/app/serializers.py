"""Serialize ORM rows into API models."""
from __future__ import annotations

import json

from sqlalchemy.orm import Session

from .models import Artifact, Dataset, Run, RunInput
from .schemas import (
    ArtifactOut,
    DatasetOut,
    InputOut,
    MessageOut,
    RunOut,
    RunSummary,
)


def _loads(value: str | None) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def artifact_out(artifact: Artifact) -> ArtifactOut:
    return ArtifactOut.model_validate(artifact, from_attributes=True)


def dataset_out(dataset: Dataset) -> DatasetOut:
    return DatasetOut.model_validate(dataset, from_attributes=True)


def message_out(run: Run) -> MessageOut:
    return MessageOut(key=run.message_key, params=_loads(run.message_params_json))


def run_summary(run: Run) -> RunSummary:
    return RunSummary(
        id=run.id,
        module=run.module,
        task_key=run.task_key,
        status=run.status,
        progress=run.progress,
        message=message_out(run),
        created_at=run.created_at,
        finished_at=run.finished_at,
    )


def _input_filename(db: Session, ref: RunInput) -> str:
    if ref.dataset_id:
        dataset = db.get(Dataset, ref.dataset_id)
        return dataset.original_filename if dataset else "dataset"
    if ref.artifact_id:
        artifact = db.get(Artifact, ref.artifact_id)
        return artifact.logical_name if artifact else "artifact"
    return "input"


def run_out(db: Session, run: Run) -> RunOut:
    inputs = [
        InputOut(
            role=ref.role,
            ordinal=ref.ordinal,
            dataset_id=ref.dataset_id,
            artifact_id=ref.artifact_id,
            filename=_input_filename(db, ref),
        )
        for ref in sorted(run.inputs, key=lambda item: (item.role, item.ordinal))
    ]
    artifacts = sorted(run.artifacts, key=lambda item: (not item.is_primary, item.logical_name))
    return RunOut(
        **run_summary(run).model_dump(),
        stage_key=run.stage_key,
        error_text=run.error_text,
        params=_loads(run.params_json),
        params_version=run.params_version,
        started_at=run.started_at,
        inputs=inputs,
        artifacts=[artifact_out(item) for item in artifacts],
    )
