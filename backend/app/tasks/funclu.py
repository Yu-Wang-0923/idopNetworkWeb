"""FunClu tasks: EM functional clustering (K) and the BIC scan."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.jobs.context import TaskContext
from app.jobs.registry import StageSpec, TaskSpec, register
from app.pure.charts import funclu_html
from app.pure.funclu import (
    build_funclu_k_export_zip,
    finite_condition_frames,
    load_curve_fitting_zip,
    save_bic_comparison,
    scan_bic,
)
from app.tasks.common import write_text


class FunCluKParams(BaseModel):
    n_components: int = Field(default=3, ge=2, le=10000)
    max_iter: int = Field(default=1000, ge=10, le=2000)
    tol: float = Field(default=1e-4, ge=1e-10, le=1.0)
    random_state: int = Field(default=42, ge=0, le=9999)
    covariance_model: Literal["S1", "S2", "S3", "S4"] = "S1"
    init_params: Literal["log_c_beta", "kmeans", "random"] = "log_c_beta"
    use_minibatch_kmeans: bool = False


class FunCluBicParams(BaseModel):
    k_min: int = Field(default=2, ge=1, le=10000)
    k_max: int = Field(default=20, ge=1, le=10000)
    step: int = Field(default=1, ge=1, le=100)
    covariance_models: list[Literal["S1", "S2", "S3", "S4"]] = Field(
        default_factory=lambda: ["S1"]
    )
    init_params: Literal["log_c_beta", "kmeans", "random"] = "log_c_beta"
    use_minibatch_kmeans: bool = False
    n_init: int = Field(default=5, ge=1, le=10)
    max_iter: int = Field(default=1000, ge=10, le=2000)
    tol: float = Field(default=1e-4, ge=1e-10, le=1.0)
    random_state: int = Field(default=42, ge=0, le=9999)
    min_cluster_size: int = Field(default=1, ge=1, le=10000)


K_STAGES = (
    StageSpec("funclu.stage.load", 5),
    StageSpec("funclu.stage.fit", 75),
    StageSpec("funclu.stage.export", 10),
    StageSpec("funclu.stage.charts", 10),
)

BIC_STAGES = (
    StageSpec("funclu.stage.load", 5),
    StageSpec("funclu.stage.bic", 85),
    StageSpec("funclu.stage.export", 10),
)


def _curve_zip_input(ctx: TaskContext) -> Path:
    inputs = [item for item in ctx.inputs if item.role == "curve_zip"]
    if len(inputs) != 1:
        raise ValueError("Exactly one Curve Fitting export ZIP is required")
    item = inputs[0]
    if Path(item.path).suffix.lower() != ".zip":
        raise ValueError("The FunClu input must be a Curve Fitting export ZIP")
    return Path(item.path)


def _prepare(ctx: TaskContext, path: Path):
    curve_sample, quasi_dynamic = load_curve_fitting_zip(path.read_bytes())
    conditions = list(curve_sample)
    frames, valid_columns = finite_condition_frames(
        [curve_sample[name] for name in conditions]
    )
    features = len(valid_columns)
    if features < 2:
        raise ValueError("FunClu needs at least 2 shared finite features")
    ctx.log(
        "funclu.message.loaded",
        conditions=len(conditions),
        features=features,
    )
    return curve_sample, quasi_dynamic, conditions, frames, features


def _finite(value) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def bic_boundary_side(k_values: list[int], best_k: int, n_features: int) -> str | None:
    """Name the end of the scanned K range that the recommendation sits on.

    Returns ``"upper"`` or ``"lower"`` only when the scan could be widened on
    that side (K stays within 1..n_features), otherwise ``None``.
    """
    if len(k_values) < 2:
        return None
    if best_k == max(k_values) and best_k < n_features:
        return "upper"
    if best_k == min(k_values) and best_k > 1:
        return "lower"
    return None


def _parameter_tables(model, conditions: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mu = np.asarray(model.params_mu, dtype=float)
    cov = np.asarray(model.params_cov, dtype=float)
    mu_rows = [
        {
            "cluster": f"M{cluster + 1}",
            "condition": conditions[condition],
            "a": float(mu[cluster, condition, 0]),
            "b": float(mu[cluster, condition, 1]),
        }
        for cluster in range(model.n_components)
        for condition in range(model.n_conditions)
    ]
    cov_rows = [
        {
            "cluster": f"M{cluster + 1}",
            "condition": conditions[condition],
            "phi": float(cov[cluster, condition, 0]),
            "gamma": float(cov[cluster, condition, 1]),
        }
        for cluster in range(model.n_components)
        for condition in range(model.n_conditions)
    ]
    return pd.DataFrame(mu_rows), pd.DataFrame(cov_rows)


def _render_cluster_profiles(
    curve_sample: dict[str, pd.DataFrame],
    labels: np.ndarray,
    common_cols: list[str],
    n_components: int,
    conditions: list[str],
) -> tuple[str, int]:
    frames = {
        name: frame.loc[:, [column for column in common_cols if column in frame.columns]]
        for name, frame in curve_sample.items()
    }
    cap = 40
    html = ""
    for _ in range(4):
        panels = funclu_html.build_cluster_panels(
            frames,
            labels,
            common_cols,
            n_components,
            show_members=True,
            show_ci=True,
            max_members=cap,
        )
        html = funclu_html.render(panels, conditions=conditions)
        if len(html.encode("utf-8")) <= 2_000_000 or cap <= 5:
            return html, cap
        cap = max(5, cap // 2)
    return html, cap


def _run_funclu_k(ctx: TaskContext) -> None:
    params = FunCluKParams.model_validate(ctx.params)
    path = _curve_zip_input(ctx)
    ctx.check_cancelled()
    ctx.progress("funclu.stage.load", 0.03, "funclu.message.loading", name=path.name)
    curve_sample, quasi_dynamic, conditions, frames, features = _prepare(ctx, path)
    if params.n_components > features:
        raise ValueError(
            f"n_components={params.n_components} exceeds the {features} shared features"
        )

    from app.pure.fast_funclu_power import PowerFunClu

    ctx.check_cancelled()
    ctx.progress(
        "funclu.stage.fit",
        0.10,
        "funclu.message.fitting",
        k=params.n_components,
    )
    model = PowerFunClu(
        n_components=params.n_components,
        max_iter=params.max_iter,
        tol=params.tol,
        random_state=params.random_state,
        covariance_model=params.covariance_model,
        init_params=params.init_params,
        use_minibatch_kmeans=params.use_minibatch_kmeans,
    )
    model.fit(frames)
    ctx.check_cancelled()

    labels = np.asarray(model.labels, dtype=int)
    common_cols = list(model.common_cols)
    ctx.progress("funclu.stage.export", 0.80, "funclu.message.exporting")
    export = build_funclu_k_export_zip(
        model=model,
        cond_names=conditions,
        curve_sample_dict=curve_sample,
        quasi_dynamic_dict=quasi_dynamic,
    )
    export_path = ctx.output_dir / "funclu_k_export.zip"
    export_path.write_bytes(export)
    ctx.add_artifact(
        export_path,
        "funclu_k_export.zip",
        "funclu_k_export",
        kind="zip",
        mime="application/zip",
        is_primary=True,
    )

    labels_df = pd.DataFrame(
        {
            "feature": common_cols,
            "cluster_id": labels + 1,
            "cluster": [f"M{k + 1}" for k in labels],
        }
    )
    sizes_df = (
        labels_df.groupby(["cluster_id", "cluster"], as_index=False)
        .size()
        .rename(columns={"size": "n_features"})
        .sort_values("cluster_id")
    )
    mu_df, cov_df = _parameter_tables(model, conditions)
    for frame, name in (
        (labels_df, "labels.csv"),
        (sizes_df, "cluster_sizes.csv"),
        (mu_df, "mu_params.csv"),
        (cov_df, "cov_params.csv"),
    ):
        artifact_path = write_text(
            ctx.output_dir / "tables" / name, frame.to_csv(index=False)
        )
        ctx.add_artifact(
            artifact_path,
            f"tables/{name}",
            "funclu_table",
            kind="csv",
            mime="text/csv",
        )

    ctx.check_cancelled()
    ctx.progress("funclu.stage.charts", 0.92, "funclu.message.charting")
    html, member_cap = _render_cluster_profiles(
        curve_sample, labels, common_cols, params.n_components, conditions
    )
    chart_path = write_text(
        ctx.output_dir / "charts" / "cluster_profiles.html", html
    )
    ctx.add_artifact(
        chart_path,
        "charts/cluster_profiles.html",
        "funclu_cluster_profiles_html",
        kind="html",
        mime="text/html",
    )

    weights = np.asarray(model.weights, dtype=float)
    summary = {
        "task": "funclu.k",
        "conditions": conditions,
        "n_conditions": len(conditions),
        "n_features": int(model.n_features),
        "n_components": int(model.n_components),
        "covariance_model": params.covariance_model,
        "covariance_spec": model.covariance_spec(),
        "init_params": params.init_params,
        "init_params_used": model.init_params_used,
        "init_fallback_reason": model.init_fallback_reason,
        "use_minibatch_kmeans": bool(params.use_minibatch_kmeans),
        "converged": bool(model.converged),
        "n_iter_run": int(model.n_iter_run),
        "max_iter": int(params.max_iter),
        "tol": float(params.tol),
        "random_state": int(params.random_state),
        "log_likelihood": float(model.log_likelihood),
        "bic": float(model.bic),
        "cluster_sizes": {
            f"M{index + 1}": int(size)
            for index, size in enumerate(np.bincount(labels, minlength=params.n_components))
        },
        "cluster_weights": {
            f"M{index + 1}": float(weight)
            for index, weight in enumerate(weights)
        },
        "chart_member_cap": member_cap,
    }
    summary_path = write_text(
        ctx.output_dir / "funclu_k_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    ctx.add_artifact(
        summary_path,
        "funclu_k_summary.json",
        "funclu_k_summary",
        kind="json",
        mime="application/json",
    )
    ctx.progress(
        "funclu.stage.charts",
        1.0,
        "funclu.message.done",
        k=params.n_components,
        clusters=params.n_components,
    )


def _run_funclu_bic(ctx: TaskContext) -> None:
    params = FunCluBicParams.model_validate(ctx.params)
    path = _curve_zip_input(ctx)
    ctx.check_cancelled()
    ctx.progress("funclu.stage.load", 0.02, "funclu.message.loading", name=path.name)
    curve_sample, quasi_dynamic, conditions, frames, features = _prepare(ctx, path)

    k_min = params.k_min
    k_max = min(params.k_max, features)
    if k_max < k_min:
        raise ValueError(
            f"k_max={k_max} is below k_min={k_min} for {features} shared features"
        )

    def on_progress(k: int, index: int, total: int) -> None:
        ctx.progress(
            "funclu.stage.bic",
            0.05 + 0.80 * index / max(1, total),
            "funclu.message.bic",
            k=k,
            index=index,
            total=total,
        )

    ctx.check_cancelled()
    bic_df = scan_bic(
        frames,
        k_min=k_min,
        k_max=k_max,
        step=params.step,
        covariance_models=tuple(params.covariance_models),
        init_params=params.init_params,
        use_minibatch_kmeans=params.use_minibatch_kmeans,
        n_init=params.n_init,
        max_iter=params.max_iter,
        tol=params.tol,
        random_state=params.random_state,
        min_cluster_size=params.min_cluster_size,
        progress=on_progress,
        check=ctx.check_cancelled,
    )
    ctx.check_cancelled()

    selected = bic_df[bic_df["selected"]].iloc[0]
    best_k = int(selected["K"])
    selected_model = str(selected["covariance_model"])
    valid = bic_df.dropna(subset=["BIC"])
    raw_best_row = valid.loc[valid["BIC"].idxmin()]
    raw_best = int(raw_best_row["K"])
    raw_best_model = str(raw_best_row["covariance_model"])
    eligible_count = int(bic_df["eligible"].sum())
    min_size_violated = eligible_count == 0
    k_values = sorted(int(k) for k in bic_df["K"].unique())
    boundary_side = bic_boundary_side(k_values, best_k, features)
    if boundary_side is not None:
        ctx.log(
            f"funclu.message.bic_boundary_{boundary_side}",
            level="warning",
            k=best_k,
        )
    if min_size_violated:
        ctx.log(
            "funclu.message.bic_min_size_violated",
            level="warning",
            min_size=params.min_cluster_size,
        )

    ctx.progress("funclu.stage.export", 0.90, "funclu.message.exporting")
    csv_path = write_text(
        ctx.output_dir / "funclu_bic_results.csv", bic_df.to_csv(index=False)
    )
    ctx.add_artifact(
        csv_path,
        "funclu_bic_results.csv",
        "funclu_bic_results",
        kind="csv",
        mime="text/csv",
        is_primary=True,
    )
    comparison_path = save_bic_comparison(
        bic_df,
        ctx.output_dir / "charts" / "bic_comparison.png",
    )
    ctx.add_artifact(
        comparison_path,
        "charts/bic_comparison.png",
        "funclu_bic_comparison",
        kind="png",
        mime="image/png",
    )
    best_row = {
        "K": best_k,
        "covariance_model": selected_model,
        "BIC": _finite(selected["BIC"]),
        "AIC": _finite(selected["AIC"]),
        "delta_bic": _finite(selected["delta_bic"]),
        "SI": _finite(selected["SI"]),
        "converged": bool(selected["converged"]),
        "min_cluster_size": int(selected["min_cluster_size"]),
        "n_parameters": int(selected["n_parameters"]),
        "n_covariance_parameters": int(selected["n_covariance_parameters"]),
        "init_params_used": str(selected["init_params_used"]),
    }
    summary = {
        "task": "funclu.bic",
        "conditions": conditions,
        "n_conditions": len(conditions),
        "n_features": features,
        "k_min": k_min,
        "k_max": k_max,
        "step": params.step,
        "n_init": params.n_init,
        "covariance_models": list(params.covariance_models),
        "init_params": params.init_params,
        "use_minibatch_kmeans": bool(params.use_minibatch_kmeans),
        "k_values": k_values,
        "recommended_k": best_k,
        "selected_model": selected_model,
        "raw_best_k": raw_best,
        "raw_best_model": raw_best_model,
        "selected_at_boundary": boundary_side is not None,
        "boundary_side": boundary_side,
        "eligible_count": eligible_count,
        "min_size_violated": min_size_violated,
        "best_row": best_row,
        "rows": int(len(bic_df)),
        "params": params.model_dump(),
    }
    summary_path = write_text(
        ctx.output_dir / "funclu_bic_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    ctx.add_artifact(
        summary_path,
        "funclu_bic_summary.json",
        "funclu_bic_summary",
        kind="json",
        mime="application/json",
    )
    ctx.progress(
        "funclu.stage.export",
        1.0,
        "funclu.message.bic_done",
        best=best_k,
        model=selected_model,
        k_count=len(summary["k_values"]),
        model_count=len(params.covariance_models),
    )


register(
    TaskSpec(
        key="funclu.k",
        module="funclu",
        title_key="tasks.funclu_k.title",
        description_key="tasks.funclu_k.description",
        params_model=FunCluKParams,
        input_roles=("curve_zip",),
        stages=K_STAGES,
        maturity="stable",
        enabled=True,
        timeout_s=3600,
        run=_run_funclu_k,
    )
)

register(
    TaskSpec(
        key="funclu.bic",
        module="funclu",
        title_key="tasks.funclu_bic.title",
        description_key="tasks.funclu_bic.description",
        params_model=FunCluBicParams,
        input_roles=("curve_zip",),
        stages=BIC_STAGES,
        maturity="stable",
        enabled=True,
        timeout_s=7200,
        run=_run_funclu_bic,
    )
)
