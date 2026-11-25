"""NetRecon single-layer network reconstruction (ASGL / static / dynamic)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from app.jobs.context import TaskContext
from app.jobs.registry import StageSpec, TaskSpec, register
from app.pure.charts import netrecon_html
from app.pure.netrecon import (
    build_effect_zip,
    build_network_html,
    build_singlelayer_export_zip,
    collect_network_export_artifacts,
    finite_network_frames,
    fit_single_network,
    load_curve_maps,
)
from app.tasks.common import write_text


class NetReconSingleParams(BaseModel):
    algorithm: Literal["asgl", "static_idop", "dynamic_idop"] = "asgl"
    condition: str | None = None
    max_order: int = Field(default=1, ge=1, le=100)
    nonneg_self: bool = True
    max_interactions: int = Field(default=0, ge=0, le=100)
    adjacency_aggregation: Literal["mean", "integral", "energy_share"] = "mean"
    lasso_alpha: float = Field(default=0.1, gt=0, le=1)
    lasso_windows: int = Field(default=1, ge=0, le=50)
    lasso_threshold: float = Field(default=0.5, ge=0, le=1)
    n_fourier: int = Field(default=0, ge=0, le=2000)
    dynamic_window: int = Field(default=0, ge=0, le=1_000_000)
    dynamic_fs: float = Field(default=100.0, ge=1, le=10000)
    top_edges: int = Field(default=0, ge=0, le=100_000)


STAGES = (
    StageSpec("netrecon.stage.load", 5),
    StageSpec("netrecon.stage.params", 10),
    StageSpec("netrecon.stage.fit", 55),
    StageSpec("netrecon.stage.predict", 10),
    StageSpec("netrecon.stage.adjacency", 5),
    StageSpec("netrecon.stage.charts", 10),
    StageSpec("netrecon.stage.export", 5),
)


def _curve_zip_input(ctx: TaskContext) -> Path:
    inputs = [item for item in ctx.inputs if item.role == "curve_zip"]
    if len(inputs) != 1:
        raise ValueError("Exactly one Curve Fitting export ZIP is required")
    path = Path(inputs[0].path)
    if path.suffix.lower() != ".zip":
        raise ValueError("The NetRecon input must be a Curve Fitting export ZIP")
    return path


def _preview(names: list[str], limit: int = 20) -> str:
    shown = ", ".join(names[:limit])
    return shown + (", ..." if len(names) > limit else "")


def _intercept_series(model, columns: list[str]) -> pd.Series:
    try:
        intercept = model.coef_.loc["intercept"]
        return intercept.reindex(columns).fillna(0.0)
    except Exception:
        return pd.Series(0.0, index=columns)


def _run_netrecon_single(ctx: TaskContext) -> None:
    params = NetReconSingleParams.model_validate(ctx.params)
    path = _curve_zip_input(ctx)
    ctx.check_cancelled()
    ctx.progress("netrecon.stage.load", 0.03, "netrecon.message.loading", name=path.name)
    quasi_map, curve_map = load_curve_maps(path.read_bytes())
    conditions = sorted(set(quasi_map) & set(curve_map))
    if not conditions:
        raise ValueError("ZIP has no condition with both quasi_dynamic and curve_sample")
    condition = params.condition or conditions[0]
    if condition not in conditions:
        raise ValueError(
            f"Condition {condition!r} is not present; available: {', '.join(conditions)}"
        )
    curve_sample, quasi_dynamic, valid = finite_network_frames(
        curve_map[condition], quasi_map[condition]
    )
    dropped = [
        str(column)
        for column in curve_map[condition].columns
        if str(column) not in set(valid)
    ]
    if dropped:
        ctx.log(
            "netrecon.message.features_dropped",
            level="warning",
            count=len(dropped),
            features=_preview(dropped),
        )

    aggregation = params.adjacency_aggregation
    if params.algorithm == "dynamic_idop":
        aggregation = "energy_share"
    ctx.progress(
        "netrecon.stage.params",
        0.12,
        "netrecon.message.params",
        algorithm=params.algorithm,
        features=len(valid),
    )
    ctx.check_cancelled()
    ctx.progress(
        "netrecon.stage.fit",
        0.25,
        "netrecon.message.fitting",
        algorithm=params.algorithm,
        max_order=params.max_order,
    )
    result = fit_single_network(
        curve_sample,
        quasi_dynamic,
        algorithm=params.algorithm,
        max_order=params.max_order,
        nonneg_self=params.nonneg_self,
        max_interactions=params.max_interactions,
        adjacency_aggregation=aggregation,
        lasso_alpha=params.lasso_alpha,
        lasso_windows=params.lasso_windows,
        lasso_threshold=params.lasso_threshold,
        n_fourier=params.n_fourier,
        dynamic_window=params.dynamic_window,
        dynamic_fs=params.dynamic_fs,
    )
    result["condition"] = condition
    model = result["model"]
    ctx.check_cancelled()
    ctx.progress("netrecon.stage.predict", 0.78, "netrecon.message.predicting")
    predicted = result["predicted_df"]
    effects = result["effect_df_list"]
    adjacency = result["adj_df"]
    ctx.progress(
        "netrecon.stage.adjacency",
        0.84,
        "netrecon.message.adjacency",
        nodes=int(adjacency.shape[0]),
    )

    ctx.check_cancelled()
    charts_dir = ctx.output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    intercept = _intercept_series(model, [str(c) for c in quasi_dynamic.columns])
    subtitle = (
        f"{condition} · {params.algorithm} · {aggregation}"
        f" · max_order={params.max_order}"
    )
    ctx.progress("netrecon.stage.charts", 0.87, "netrecon.message.charting", name="effect decomposition")
    effect_html = netrecon_html.render_effects(
        predicted,
        effects,
        intercept,
        title="Effect decomposition",
        subtitle=subtitle,
    )
    effect_path = write_text(charts_dir / "effect_decomposition.html", effect_html)
    ctx.add_artifact(
        effect_path,
        "charts/effect_decomposition.html",
        "netrecon_effect_html",
        kind="html",
        mime="text/html",
    )
    ctx.progress("netrecon.stage.charts", 0.92, "netrecon.message.charting", name="adjacency heatmap")
    heatmap_html = netrecon_html.render_heatmap(
        adjacency,
        title="Adjusted Matrix Heatmap",
        subtitle=subtitle,
    )
    heatmap_path = write_text(charts_dir / "adjacency_heatmap.html", heatmap_html)
    ctx.add_artifact(
        heatmap_path,
        "charts/adjacency_heatmap.html",
        "netrecon_heatmap_html",
        kind="html",
        mime="text/html",
    )
    ctx.progress("netrecon.stage.charts", 0.96, "netrecon.message.charting", name="interactive network")
    network_path = build_network_html(
        adjacency, charts_dir, top_edges=params.top_edges
    )
    ctx.add_artifact(
        network_path,
        "charts/network.html",
        "netrecon_network_html",
        kind="html",
        mime="text/html",
    )
    for name in ("13A_ring_layout.csv", "13B_display_edges.csv"):
        source = charts_dir / name
        if source.exists():
            ctx.add_artifact(
                source,
                name,
                "netrecon_layout_table",
                kind="csv",
                mime="text/csv",
            )

    ctx.check_cancelled()
    ctx.progress("netrecon.stage.export", 0.98, "netrecon.message.exporting")
    export_bytes = build_singlelayer_export_zip(result)
    export_path = ctx.output_dir / "single_layer_idopnetwork_export.zip"
    export_path.write_bytes(export_bytes)
    ctx.add_artifact(
        export_path,
        "single_layer_idopnetwork_export.zip",
        "single_layer_idopnetwork_export",
        kind="zip",
        mime="application/zip",
        is_primary=True,
    )
    artifacts = collect_network_export_artifacts(result)
    effect_zip = build_effect_zip(artifacts["effect_map"])
    effect_zip_path = ctx.output_dir / "effect_decomposition.zip"
    effect_zip_path.write_bytes(effect_zip)
    ctx.add_artifact(
        effect_zip_path,
        "effect_decomposition.zip",
        "netrecon_effect_tables",
        kind="zip",
        mime="application/zip",
    )
    for frame, name in (
        (adjacency, "adjacency_matrix.csv"),
        (artifacts["from_to_df"], "from_to.csv"),
        (predicted, "predictions.csv"),
        (result["quasi_dynamic_df"], "observations.csv"),
    ):
        table_path = write_text(
            ctx.output_dir / "tables" / name, frame.to_csv(index=True)
        )
        ctx.add_artifact(
            table_path,
            f"tables/{name}",
            "netrecon_table",
            kind="csv",
            mime="text/csv",
        )
    summary = {
        "task": "netrecon.single",
        "conditions": conditions,
        "condition": condition,
        "algorithm": params.algorithm,
        "adjacency_aggregation": aggregation,
        "n_features": len(valid),
        "n_nodes": int(adjacency.shape[0]),
        "n_edges_nonzero": int(artifacts["from_to_df"].shape[0]),
        "top_edges": int(params.top_edges),
        "display_edges": int(
            len(pd.read_csv(charts_dir / "13B_display_edges.csv"))
            if (charts_dir / "13B_display_edges.csv").exists()
            else 0
        ),
        "mse": float(model.mse_) if model.mse_ is not None else None,
        "selected_max_order": int(model.max_order),
        "alpha": float(model.alpha),
        "lasso_alpha": float(params.lasso_alpha),
        "lasso_windows": int(params.lasso_windows),
        "lasso_threshold": float(params.lasso_threshold),
        "excluded_features": dropped,
        "adaptation_notes": list(getattr(model, "adaptation_notes_", [])),
        "solver_version": getattr(model, "solver_version_", None),
        "dynamic_settings": (
            {
                "n_fourier": int(getattr(model, "n_fourier", params.n_fourier)),
                "fs": float(params.dynamic_fs),
                "window": params.dynamic_window,
            }
            if params.algorithm == "dynamic_idop"
            else None
        ),
        "rows": int(result["quasi_dynamic_df"].shape[0]),
        "params": params.model_dump(),
    }
    summary_path = write_text(
        ctx.output_dir / "netrecon_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    ctx.add_artifact(
        summary_path,
        "netrecon_summary.json",
        "netrecon_summary",
        kind="json",
        mime="application/json",
    )
    ctx.progress(
        "netrecon.stage.export",
        1.0,
        "netrecon.message.done",
        condition=condition,
        nodes=int(adjacency.shape[0]),
        edges=int(artifacts["from_to_df"].shape[0]),
    )


register(
    TaskSpec(
        key="netrecon.single",
        module="netrecon",
        title_key="tasks.netrecon_single.title",
        description_key="tasks.netrecon_single.description",
        params_model=NetReconSingleParams,
        input_roles=("curve_zip",),
        stages=STAGES,
        maturity="stable",
        enabled=True,
        timeout_s=7200,
        run=_run_netrecon_single,
    )
)
