"""Static task registry shared by the API and the worker."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .context import TaskContext

if TYPE_CHECKING:
    from app.config import Settings


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


def task_enabled(spec: TaskSpec, settings: "Settings | None" = None) -> bool:
    if not spec.enabled:
        return False
    return settings is None or spec.key not in settings.disabled_task_keys


def task_maturity(spec: TaskSpec, settings: "Settings | None" = None) -> str:
    if settings is not None and spec.key in settings.beta_task_keys:
        return "beta"
    return spec.maturity


def enabled_tasks(settings: "Settings | None" = None) -> list[TaskSpec]:
    return [spec for spec in TASKS.values() if task_enabled(spec, settings)]


def task_public_spec(
    spec: TaskSpec, settings: "Settings | None" = None
) -> dict[str, Any]:
    return {
        "key": spec.key,
        "module": spec.module,
        "title_key": spec.title_key,
        "description_key": spec.description_key,
        "params_schema": spec.params_model.model_json_schema(),
        "input_roles": list(spec.input_roles),
        "stages": [{"key": stage.key, "weight": stage.weight} for stage in spec.stages],
        "maturity": task_maturity(spec, settings),
        "enabled": task_enabled(spec, settings),
    }


# Import task modules to populate the registry.  Kept at the bottom to avoid a
# circular import: task modules import TaskSpec/register from this module.
from app import tasks as _tasks  # noqa: E402,F401
