"""NetAnal GLMY path-homology analysis from a NetRecon export."""
from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.jobs.context import TaskContext
from app.jobs.registry import StageSpec, TaskSpec, register
from app.tasks.common import write_text


class NetAnalParams(BaseModel):
    member: str = Field(min_length=1)
    mode: Literal["all_edges_colored", "separate_signs"] = "all_edges_colored"
    n_stages: int = Field(default=4, ge=2, le=12)
    max_columns: int = Field(default=5, ge=1, le=8)
    thresholds: list[float] | None = None
    export_vector: bool = False


STAGES = (
    StageSpec("netanal.stage.load", 5),
    StageSpec("netanal.stage.compute", 80),
    StageSpec("netanal.stage.save", 10),
    StageSpec("netanal.stage.pack", 5),
)

_EXPORT_SUFFIXES = {
    ".png": ("png", "image/png"),
    ".svg": ("svg", "image/svg+xml"),
    ".pdf": ("pdf", "application/pdf"),
    ".csv": ("csv", "text/csv"),
    ".json": ("json", "application/json"),
}


def _netrecon_zip_input(ctx: TaskContext) -> Path:
    inputs = [item for item in ctx.inputs if item.role == "netrecon_zip"]
    if len(inputs) != 1:
        raise ValueError("Exactly one NetRecon export ZIP is required")
    path = Path(inputs[0].path)
    if path.suffix.lower() != ".zip":
        raise ValueError("The NetAnal input must be a NetRecon export ZIP")
    return path


def _run_netanal_glmy(ctx: TaskContext) -> None:
    params = NetAnalParams.model_validate(ctx.params)
    path = _netrecon_zip_input(ctx)
    ctx.check_cancelled()
    ctx.progress(
        "netanal.stage.load", 0.02, "netanal.message.loading", name=path.name
    )

    from idopnetwork.analysis.network_analysis import (
        list_from_to_members,
        load_from_to_from_zip,
    )
    from idopnetwork.analysis.regular_glmy.workflow import run_regular_glmy

    zip_bytes = path.read_bytes()
    members = list_from_to_members(zip_bytes)
    if not members:
        raise ValueError("NetRecon ZIP does not contain any from_to.csv")
    if params.member not in members:
        raise ValueError(
            f"Member {params.member!r} is not present; available: {', '.join(members)}"
        )
    frame = load_from_to_from_zip(zip_bytes, params.member)
    if frame.empty:
        raise ValueError("The selected network has no edges")
    nodes = set(frame["from"]) | set(frame["to"])
    ctx.log(
        "netanal.message.loaded",
        member=params.member,
        edges=len(frame),
        nodes=len(nodes),
    )
    ctx.check_cancelled()
    ctx.progress(
        "netanal.stage.compute",
        0.08,
        "netanal.message.computing",
        mode=params.mode,
    )
    result = run_regular_glmy(
        frame,
        mode=params.mode,
        n_stages=params.n_stages,
        max_columns=params.max_columns,
        thresholds=params.thresholds,
        export_vector=params.export_vector,
    )
    ctx.check_cancelled()
    if result.get("removed_self_loops"):
        ctx.log(
            "netanal.message.self_loops",
            loops=int(result["removed_self_loops"]),
        )

    ctx.progress("netanal.stage.save", 0.90, "netanal.message.saving")
    artifacts = result["artifacts"]
    root = ctx.output_dir / "glmy"
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in artifacts.items():
        destination = ctx.output_dir / "glmy" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        kind, mime = _EXPORT_SUFFIXES.get(
            destination.suffix.lower(),
            ("bin", mimetypes.guess_type(destination.name)[0] or "application/octet-stream"),
        )
        ctx.add_artifact(
            destination,
            f"glmy/{relative}",
            f"netanal_glmy_{kind}",
            kind=kind,
            mime=mime,
        )
    input_table = write_text(
        ctx.output_dir / "tables" / "from_to.csv", frame.to_csv(index=False)
    )
    ctx.add_artifact(
        input_table,
        "tables/from_to.csv",
        "netanal_table",
        kind="csv",
        mime="text/csv",
    )

    ctx.progress("netanal.stage.pack", 0.96, "netanal.message.packing")
    archive_path = ctx.output_dir / f"glmy_{params.mode}.zip"
    archive_path.write_bytes(result["zip"])
    ctx.add_artifact(
        archive_path,
        f"glmy_{params.mode}.zip",
        "netanal_glmy_zip",
        kind="zip",
        mime="application/zip",
        is_primary=True,
    )
    summary = {
        "task": "netanal.glmy",
        "member": params.member,
        "mode": params.mode,
        "n_stages": params.n_stages,
        "max_columns": params.max_columns,
        "thresholds": params.thresholds,
        "export_vector": bool(params.export_vector),
        "edges": int(len(frame)),
        "nodes": int(len(nodes)),
        "removed_self_loops": int(result.get("removed_self_loops", 0)),
        "n_nodes": int(result.get("n_nodes", len(nodes))),
        "artifact_count": len(artifacts),
        "artifacts": sorted(artifacts),
        "zip_bytes": len(result["zip"]),
        "params": params.model_dump(),
    }
    summary_path = write_text(
        ctx.output_dir / "netanal_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    ctx.add_artifact(
        summary_path,
        "netanal_summary.json",
        "netanal_summary",
        kind="json",
        mime="application/json",
    )
    ctx.progress(
        "netanal.stage.pack",
        1.0,
        "netanal.message.done",
        mode=params.mode,
        artifacts=len(artifacts),
    )


register(
    TaskSpec(
        key="netanal.glmy",
        module="netanal",
        title_key="tasks.netanal_glmy.title",
        description_key="tasks.netanal_glmy.description",
        params_model=NetAnalParams,
        input_roles=("netrecon_zip",),
        stages=STAGES,
        maturity="stable",
        enabled=True,
        timeout_s=7200,
        run=_run_netanal_glmy,
    )
)
