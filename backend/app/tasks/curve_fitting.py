"""Curve Fitting task: quasi-dynamic power fitting or SWT denoising."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from app.jobs.context import TaskContext
from app.jobs.registry import StageSpec, TaskSpec, register
from app.pure.charts import curve_fitting_html, ecg_fitting_html
from app.pure.curve_fitting import build_fitting_export, fit_uploaded_csv


class CurveFittingParams(BaseModel):
    data_mode: Literal["quasi_dynamic", "dynamic"] = "quasi_dynamic"
    first_transform: Literal["None", "Z_min_add1", "Log10_1p", "Minmax_0_1"] = "Z_min_add1"
    second_transform: Literal["None", "Z_min_add1", "Log10_1p", "Minmax_0_1"] = "Log10_1p"
    n_samples: int = Field(default=30, ge=2, le=10000)
    trim_percent: float = Field(default=1.0, ge=0, lt=100)
    swt_wavelet: Literal["rbio3.9", "db4", "sym8", "coif3", "bior3.9"] = "rbio3.9"
    swt_level: int = Field(default=2, ge=1, le=10)
    swt_alpha: float = Field(default=0.5, ge=0, le=5)
    ecg_sampling_rate: float = Field(default=100.0, ge=1, le=10000)


STAGES = (
    StageSpec("curve_fitting.stage.validate", 5),
    StageSpec("curve_fitting.stage.fit", 65),
    StageSpec("curve_fitting.stage.export", 10),
    StageSpec("curve_fitting.stage.charts", 20),
)


def _normalize_condition(name: str) -> str:
    return name.replace("\\", "_").replace("/", "_").strip() or "export"


def _filebase(name: str) -> str:
    return Path(name).stem or "data"


def _finite_feature_count(samples) -> int:
    values = samples.to_numpy(dtype=float)
    return int(np.isfinite(values).all(axis=0).sum())


def _finite_features(quasi, samples) -> list[str]:
    values = samples.to_numpy(dtype=float)
    mask = np.isfinite(values).all(axis=0)
    return [str(column) for column, keep in zip(quasi.columns, mask) if keep]


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _run_curve_fitting(ctx: TaskContext) -> None:
    params = CurveFittingParams.model_validate(ctx.params)
    csv_inputs = sorted(
        (item for item in ctx.inputs if item.role == "csv"),
        key=lambda item: item.ordinal,
    )
    if not csv_inputs:
        raise ValueError("At least one CSV input is required")
    if len(csv_inputs) > 50:
        raise ValueError("At most 50 CSV files are supported per run")

    names = [item.filename for item in csv_inputs]
    normalized = [_normalize_condition(name) for name in names]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate file names after path normalization")

    ctx.progress(
        "curve_fitting.stage.validate",
        0.02,
        "curve_fitting.message.validating",
        files=len(csv_inputs),
    )

    results: dict[str, dict] = {}
    failures: list[dict] = []
    summary_files: list[dict] = []

    for index, item in enumerate(csv_inputs):
        ctx.check_cancelled()
        fraction = 0.05 + 0.65 * index / len(csv_inputs)
        ctx.progress(
            "curve_fitting.stage.fit",
            fraction,
            "curve_fitting.message.fitting",
            index=index + 1,
            total=len(csv_inputs),
            name=item.filename,
        )
        if Path(item.path).suffix.lower() != ".csv":
            message = "Input is not a CSV file"
            failures.append({"name": item.filename, "error": message})
            ctx.log(
                "curve_fitting.message.file_failed",
                level="error",
                name=item.filename,
                raw_text=f"{item.filename}: {message}",
            )
            continue
        try:
            tables = fit_uploaded_csv(
                Path(item.path).read_bytes(),
                data_mode=params.data_mode,
                first_transform=params.first_transform,
                second_transform=params.second_transform,
                n_samples=params.n_samples,
                trim_percent=params.trim_percent,
                swt_wavelet=params.swt_wavelet,
                swt_level=params.swt_level,
                swt_alpha=params.swt_alpha,
            )
        except Exception as exc:  # noqa: BLE001 - per-file failures are reported, not fatal
            failures.append({"name": item.filename, "error": str(exc)})
            ctx.log(
                "curve_fitting.message.file_failed",
                level="error",
                name=item.filename,
                raw_text=f"{item.filename}: {exc}",
            )
            continue
        results[item.filename] = tables
        quasi = tables["quasi_dynamic"]
        samples = tables["curve_sample"]
        summary_files.append(
            {
                "name": item.filename,
                "rows": int(len(quasi)),
                "features": int(quasi.shape[1]),
                "features_fitted": _finite_feature_count(samples),
                "sample_points": int(len(samples)),
            }
        )

    if not results:
        raise RuntimeError("No input file could be fitted")

    ctx.check_cancelled()
    ctx.progress(
        "curve_fitting.stage.export",
        0.75,
        "curve_fitting.message.exporting",
        files=len(results),
    )
    export_path = ctx.output_dir / "curve_fitting_export.zip"
    export_path.write_bytes(build_fitting_export(results))
    ctx.add_artifact(
        export_path,
        "curve_fitting_export.zip",
        "curve_fitting_export",
        kind="zip",
        mime="application/zip",
        is_primary=True,
    )

    for name, tables in results.items():
        base = _filebase(name)
        for table_key, suffix in (
            ("curve_params", "params"),
            ("curve_sample", "samples"),
            ("quasi_dynamic", "quasidynamic"),
        ):
            path = _write_text(
                ctx.output_dir / "tables" / f"{base}_{suffix}.csv",
                tables[table_key].to_csv(index=True),
            )
            ctx.add_artifact(
                path,
                f"tables/{base}_{suffix}.csv",
                "curve_fitting_table",
                kind="csv",
                mime="text/csv",
            )

    summary = {
        "data_mode": params.data_mode,
        "params": params.model_dump(),
        "files": summary_files,
        "failed": failures,
    }
    summary_path = _write_text(
        ctx.output_dir / "curve_fitting_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    ctx.add_artifact(
        summary_path,
        "curve_fitting_summary.json",
        "curve_fitting_summary",
        kind="json",
        mime="application/json",
    )

    if params.data_mode == "dynamic":
        for index, (name, tables) in enumerate(results.items()):
            ctx.check_cancelled()
            fraction = 0.80 + 0.18 * index / len(results)
            ctx.progress(
                "curve_fitting.stage.charts",
                fraction,
                "curve_fitting.message.charting",
                name=name,
            )
            base = _filebase(name)
            try:
                html = ecg_fitting_html.render(
                    tables["quasi_dynamic"],
                    tables["curve_sample"],
                    sampling_rate=params.ecg_sampling_rate,
                    title=f"ECG Fitting · {name}",
                )
            except Exception as exc:  # noqa: BLE001 - chart failures are non-fatal
                ctx.log(
                    "curve_fitting.message.file_failed",
                    level="error",
                    name=name,
                    raw_text=f"{name}: ECG figure failed: {exc}",
                )
                continue
            path = _write_text(
                ctx.output_dir / "charts" / base / "ecg_fitting.html",
                html,
            )
            ctx.add_artifact(
                path,
                f"charts/{base}/ecg_fitting.html",
                "curve_fitting_ecg_html",
                kind="html",
                mime="text/html",
            )
    else:
        for index, (name, tables) in enumerate(results.items()):
            ctx.check_cancelled()
            fraction = 0.80 + 0.12 * index / len(results)
            ctx.progress(
                "curve_fitting.stage.charts",
                fraction,
                "curve_fitting.message.charting",
                name=name,
            )
            base = _filebase(name)
            features = _finite_features(
                tables["quasi_dynamic"], tables["curve_sample"]
            )
            if not features:
                ctx.log(
                    "curve_fitting.message.file_failed",
                    level="warning",
                    name=name,
                    raw_text=f"{name}: no feature has a finite fitted curve",
                )
                continue
            panels = curve_fitting_html.build_panels(
                tables["quasi_dynamic"],
                tables["curve_sample"],
                features,
            )
            html = curve_fitting_html.render(panels)
            path = _write_text(
                ctx.output_dir / "charts" / base / "curve_fitting.html",
                html,
            )
            ctx.add_artifact(
                path,
                f"charts/{base}/curve_fitting.html",
                "curve_fitting_chart",
                kind="html",
                mime="text/html",
            )

        if len(results) > 1:
            first = next(iter(results.values()))
            common = _finite_features(first["quasi_dynamic"], first["curve_sample"])
            for tables in results.values():
                available = set(
                    _finite_features(tables["quasi_dynamic"], tables["curve_sample"])
                )
                common = [column for column in common if column in available]
            if common:
                panels = curve_fitting_html.build_compare_panels(
                    [tables["quasi_dynamic"].loc[:, common] for tables in results.values()],
                    [tables["curve_sample"].loc[:, common] for tables in results.values()],
                    list(results),
                    common,
                )
                html = curve_fitting_html.render(panels)
                path = _write_text(ctx.output_dir / "charts" / "compare.html", html)
                ctx.add_artifact(
                    path,
                    "charts/compare.html",
                    "curve_fitting_compare",
                    kind="html",
                    mime="text/html",
                )

    ctx.progress(
        "curve_fitting.stage.charts",
        1.0,
        "curve_fitting.message.done",
        files=len(results),
    )


register(
    TaskSpec(
        key="curve_fitting.fit",
        module="curve_fitting",
        title_key="tasks.curve_fitting.title",
        description_key="tasks.curve_fitting.description",
        params_model=CurveFittingParams,
        input_roles=("csv",),
        stages=STAGES,
        maturity="stable",
        enabled=True,
        timeout_s=3600,
        run=_run_curve_fitting,
    )
)
