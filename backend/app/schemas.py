"""Pydantic request/response models."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class SessionOut(BaseModel):
    created_at: datetime
    retention_days: int
    max_csv_bytes: int
    max_zip_bytes: int


class DatasetOut(BaseModel):
    id: str
    original_filename: str
    size_bytes: int
    sha256: str
    detected_kind: str
    content_type: str
    created_at: datetime


class InspectOut(BaseModel):
    kind: str
    conditions: list[str] = Field(default_factory=list)
    members: list[str] = Field(default_factory=list)
    features: int | None = None
    clusters: list[str] = Field(default_factory=list)


class InputRef(BaseModel):
    role: str
    ordinal: int = 0
    dataset_id: str | None = None
    artifact_id: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "InputRef":
        if bool(self.dataset_id) == bool(self.artifact_id):
            raise ValueError("Exactly one of dataset_id or artifact_id is required")
        return self


class InputOut(BaseModel):
    role: str
    ordinal: int
    dataset_id: str | None
    artifact_id: str | None
    filename: str


class ArtifactOut(BaseModel):
    id: str
    run_id: str
    logical_name: str
    semantic_type: str
    kind: str
    mime: str
    size_bytes: int
    sha256: str
    is_primary: bool
    created_at: datetime


class MessageOut(BaseModel):
    key: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class RunCreate(BaseModel):
    task_key: str
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: list[InputRef] = Field(default_factory=list)


class RunSummary(BaseModel):
    id: str
    module: str
    task_key: str
    status: str
    progress: int
    message: MessageOut
    created_at: datetime
    finished_at: datetime | None


class RunOut(RunSummary):
    stage_key: str | None
    error_text: str | None
    params: dict[str, Any]
    params_version: int
    started_at: datetime | None
    inputs: list[InputOut]
    artifacts: list[ArtifactOut]


class StageOut(BaseModel):
    key: str
    weight: int


class TaskOut(BaseModel):
    key: str
    module: str
    title_key: str
    description_key: str
    params_schema: dict[str, Any]
    input_roles: list[str]
    stages: list[StageOut]
    maturity: Literal["stable", "beta"]
    enabled: bool


class LogOut(BaseModel):
    seq: int
    ts: datetime
    level: str
    message_key: str | None
    message_params: dict[str, Any]
    raw_text: str | None
