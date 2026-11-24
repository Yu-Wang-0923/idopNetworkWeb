"""Static task registry shared by the API and the worker."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .context import TaskContext


@dataclass(frozen=True)
class StageSpec:
    key: str
    weight: int


@dataclass(frozen=True)
class TaskSpec:
    key: str
    module: str
    title_key: str
    description_key: str
    params_model: type[BaseModel]
    input_roles: tuple[str, ...]
    stages: tuple[StageSpec, ...]
    maturity: str
    enabled: bool
    timeout_s: int
    run: Callable[[TaskContext], None]


TASKS: dict[str, TaskSpec] = {}


def register(spec: TaskSpec) -> TaskSpec:
    if spec.key in TASKS:
        raise ValueError(f"Duplicate task key: {spec.key}")
    TASKS[spec.key] = spec
    return spec


def get_task(key: str) -> TaskSpec | None:
    return TASKS.get(key)


def enabled_tasks() -> list[TaskSpec]:
    return [spec for spec in TASKS.values() if spec.enabled]


def task_public_spec(spec: TaskSpec) -> dict[str, Any]:
    return {
        "key": spec.key,
        "module": spec.module,
        "title_key": spec.title_key,
        "description_key": spec.description_key,
        "params_schema": spec.params_model.model_json_schema(),
        "input_roles": list(spec.input_roles),
        "stages": [{"key": stage.key, "weight": stage.weight} for stage in spec.stages],
        "maturity": spec.maturity,
        "enabled": spec.enabled,
    }


# Import task modules to populate the registry.  Kept at the bottom to avoid a
# circular import: task modules import TaskSpec/register from this module.
from app import tasks as _tasks  # noqa: E402,F401
