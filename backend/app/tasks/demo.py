"""Demo task used to validate the job pipeline end to end."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.jobs.context import TaskContext
from app.jobs.registry import StageSpec, TaskSpec, register


class DemoParams(BaseModel):
    message: str = Field(default="Hello from idopNetwork Web", max_length=200)
    steps: int = Field(default=5, ge=1, le=20)
    delay_ms: int = Field(default=400, ge=0, le=5000)


def _run_demo(ctx: TaskContext) -> None:
    params = DemoParams.model_validate(ctx.params)
    started = datetime.now(timezone.utc)
    ctx.progress("demo.stage.prepare", 0.05, "demo.message.ready")

    for step in range(1, params.steps + 1):
        ctx.check_cancelled()
        time.sleep(params.delay_ms / 1000.0)
        fraction = 0.05 + 0.80 * step / params.steps
        ctx.progress(
            "demo.stage.work",
            fraction,
            "demo.message.step",
            step=step,
            total=params.steps,
        )

    ctx.check_cancelled()
    ctx.progress("demo.stage.finish", 0.9, "demo.stage.finish")

    result_path = ctx.output_dir / "result.txt"
    result_path.write_text(
        f"{params.message}\nsteps={params.steps}\n"
        f"started={started.isoformat()}\n",
        encoding="utf-8",
    )
    summary_path = ctx.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "task": "demo.echo",
                "params": params.model_dump(),
                "started_at": started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    ctx.add_artifact(
        result_path,
        "result.txt",
        "demo_text",
        kind="txt",
        mime="text/plain",
        is_primary=True,
    )
    ctx.add_artifact(
        summary_path,
        "summary.json",
        "demo_summary",
        kind="json",
        mime="application/json",
    )
    ctx.progress("demo.stage.finish", 1.0, "demo.message.done")


register(
    TaskSpec(
        key="demo.echo",
        module="demo",
        title_key="tasks.demo.title",
        description_key="tasks.demo.description",
        params_model=DemoParams,
        input_roles=(),
        stages=(
            StageSpec("demo.stage.prepare", 5),
            StageSpec("demo.stage.work", 80),
            StageSpec("demo.stage.finish", 15),
        ),
        maturity="stable",
        enabled=True,
        timeout_s=300,
        run=_run_demo,
    )
)
