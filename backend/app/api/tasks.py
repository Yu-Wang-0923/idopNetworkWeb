"""Task registry endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..config import Settings, get_settings
from ..jobs.registry import enabled_tasks, task_public_spec
from ..schemas import TaskOut

router = APIRouter()


@router.get("/tasks", response_model=list[TaskOut])
def list_tasks(settings: Settings = Depends(get_settings)) -> list[dict]:
    return [task_public_spec(spec, settings) for spec in enabled_tasks(settings)]
