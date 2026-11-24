"""Task registry endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from ..jobs.registry import enabled_tasks, task_public_spec
from ..schemas import TaskOut

router = APIRouter()


@router.get("/tasks", response_model=list[TaskOut])
def list_tasks() -> list[dict]:
    return [task_public_spec(spec) for spec in enabled_tasks()]
