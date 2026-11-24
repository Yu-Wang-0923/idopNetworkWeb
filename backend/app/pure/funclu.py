# Copied from https://github.com/Yu-Wang-0923/idopNetworkAPP@fc5dc2f5a903c7fdbdadeef903771f9852f05b8c
# packages/idopnetwork-app/src/idopnetwork_app/pages/2_FunClu.py:112-227 and :63-109
# The previous Streamlit app is frozen; this copy is the maintained source for idopNetwork Web.
"""Streamlit-free FunClu helpers: curve ZIP loading, K export, BIC summaries."""
from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


def load_curve_fitting_zip(
    content: bytes,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Read curve_sample/quasi_dynamic tables from a Curve Fitting export ZIP."""
    curve_sample: dict[str, pd.DataFrame] = {}
    quasi_dynamic: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for name in archive.namelist():
            normalized = name.replace("\\", "/")
            if normalized.endswith("/"):
                continue
            if not (
                normalized.endswith("curve_sample.csv")
                or normalized.endswith("quasi_dynamic.csv")
            ):
                continue
            condition = (
                normalized.rsplit("/", 1)[0] if "/" in normalized else normalized
            )
            with archive.open(name) as handle:
                frame = pd.read_csv(handle, index_col=0)
            if normalized.endswith("curve_sample.csv"):
                curve_sample[condition] = frame
            else:
                quasi_dynamic[condition] = frame
    if not curve_sample:
        raise ValueError("ZIP does not contain any curve_sample.csv table")
    if not quasi_dynamic:
        raise ValueError(
            "ZIP does not contain quasi_dynamic.csv, which FunClu export requires"
        )
    missing = sorted(set(curve_sample) - set(quasi_dynamic))
    if missing:
        raise ValueError(
            "ZIP is missing quasi_dynamic.csv for: " + ", ".join(missing)
        )
    return (
        dict(sorted(curve_sample.items())),
        dict(sorted(quasi_dynamic.items())),
    )


def common_feature_count(frames: list[pd.DataFrame]) -> int:
    """Number of feature columns shared by every condition."""
    if not frames:
        return 0
    common = set(str(column) for column in frames[0].columns)
    for frame in frames[1:]:
        common &= set(str(column) for column in frame.columns)
    return len(common)


def finite_condition_frames(
    frames: list[pd.DataFrame],
) -> tuple[list[pd.DataFrame], list[str]]:
    """Keep shared features whose values are finite in every condition."""
    if not frames:
        raise ValueError("FunClu requires at least one condition")
    common = [str(column) for column in frames[0].columns]
    for frame in frames[1:]:
        common = [column for column in common if column in frame.columns]
    valid = []
    for column in common:
        if all(
            np.isfinite(frame[column].to_numpy(dtype=float)).all()
            for frame in frames
        ):
            valid.append(column)
    if not valid:
        raise ValueError("No shared finite feature is available for FunClu")
    return [frame.loc[:, valid] for frame in frames], valid


def observed_condition_frames(
    quasi_dynamic: dict[str, pd.DataFrame],
    curve_sample: dict[str, pd.DataFrame],
    conditions: list[str],
) -> tuple[list[pd.DataFrame], list[str]]:
    """FunClu input: the observed quasi_dynamic tables, never the sampled power curves.

    A feature is kept only when its observations and its fitted curve are finite
    in every condition, because the K export and the figure carry both tables.
    """
    frames, observed = finite_condition_frames(
        [quasi_dynamic[name] for name in conditions]
    )
    _, fitted = finite_condition_frames([curve_sample[name] for name in conditions])
    fitted_set = set(fitted)
    valid = [column for column in observed if column in fitted_set]
    if not valid:
        raise ValueError(
            "No shared feature has finite observations and a finite fitted curve"
        )
    return [frame.loc[:, valid] for frame in frames], valid


def build_funclu_k_export_zip(
    *,
    model: Any,
    cond_names: list[str],
    curve_sample_dict: dict[str, pd.DataFrame],
    quasi_dynamic_dict: dict[str, pd.DataFrame],
) -> bytes:
    """Build the FunClu-K ZIP consumed by multi-layer NetRecon.

    Cluster centers are the fitted cluster means evaluated on each condition's
    curve_sample grid, plus the mean of the members' observations.
    """
    if model.labels is None or model.common_cols is None:
        raise ValueError("FunClu-K result is incomplete: missing labels/common columns.")

    labels_np = np.asarray(model.labels, dtype=int)
    common_cols = list(model.common_cols)
    if len(labels_np) != len(common_cols):
        raise ValueError("FunClu-K labels do not match the clustered feature columns.")
    missing_quasi = [
        name for name in cond_names if name not in quasi_dynamic_dict
    ]
    if missing_quasi:
        raise ValueError(
            "FunClu-K export requires quasi_dynamic for every condition; "
            f"missing: {', '.join(missing_quasi)}"
        )

    labels_df = pd.DataFrame(
        {
            "feature": common_cols,
            "cluster_id": labels_np + 1,
            "cluster": [f"M{k + 1}" for k in labels_np],
        }
    )
    cluster_sizes_df = (
        labels_df.groupby(["cluster_id", "cluster"], as_index=False)
        .size()
        .rename(columns={"size": "n_features"})
        .sort_values("cluster_id")
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("labels.csv", labels_df.to_csv(index=False))
        archive.writestr("cluster_sizes.csv", cluster_sizes_df.to_csv(index=False))

        for cond_idx, cond_name in enumerate(cond_names):
            curve_sample = curve_sample_dict[cond_name].loc[:, common_cols]
            quasi_dynamic = quasi_dynamic_dict[cond_name].loc[:, common_cols]

            for cluster_idx in range(model.n_components):
                cluster_name = f"M{cluster_idx + 1}"
                member_cols = labels_df.loc[
                    labels_df["cluster_id"] == cluster_idx + 1, "feature"
                ].tolist()
                archive.writestr(
                    f"cluster_members/{cond_name}/{cluster_name}_curve_sample.csv",
                    curve_sample.loc[:, member_cols].to_csv(index=True),
                )
                archive.writestr(
                    f"cluster_members/{cond_name}/{cluster_name}_quasi_dynamic.csv",
                    quasi_dynamic.loc[:, member_cols].to_csv(index=True),
                )

            center_df = pd.DataFrame(
                model.mean_curves_at(cond_idx, curve_sample.index).T,
                index=curve_sample.index,
                columns=[f"M{k + 1}" for k in range(model.n_components)],
            )
            center_df.index.name = curve_sample.index.name or "time"
            archive.writestr(
                f"cluster_centers/{cond_name}/cluster_center_curve_sample.csv",
                center_df.to_csv(index=True),
            )
            center_response_df = pd.DataFrame(index=quasi_dynamic.index)
            for cluster_idx in range(model.n_components):
                cluster_name = f"M{cluster_idx + 1}"
                member_cols = labels_df.loc[
                    labels_df["cluster_id"] == cluster_idx + 1, "feature"
                ].tolist()
                center_response_df[cluster_name] = quasi_dynamic.loc[
                    :, member_cols
                ].mean(axis=1)
            center_response_df.index.name = quasi_dynamic.index.name or "time"
            archive.writestr(
                f"cluster_centers/{cond_name}/cluster_center_quasi_dynamic.csv",
                center_response_df.to_csv(index=True),
            )
    return buffer.getvalue()


def _silhouette_index(
    data: list[pd.DataFrame], common_cols: list[str], labels: np.ndarray
) -> float:
    from sklearn.metrics import silhouette_score

    if labels.size < 3 or np.unique(labels).size < 2:
        return float("nan")
    try:
        matrix = np.concatenate(
            [
                frame.loc[:, common_cols].to_numpy(dtype=float).T
                for frame in data
            ],
            axis=1,
        )
        mean = np.nanmean(matrix, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        matrix = np.where(np.isfinite(matrix), matrix, mean)
        scale = np.nanstd(matrix, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        return float(silhouette_score((matrix - mean) / scale, labels))
    except Exception:
        return float("nan")


def scan_bic(
    data: list[pd.DataFrame],
    parameter_data: list[pd.DataFrame] | None = None,
    *,
    k_min: int = 2,
    k_max: int = 10,
    step: int = 1,
    covariance_models: tuple[str, ...] = ("S1", "S2", "S3", "S4"),
    init_params: str = "log_c_beta",
    use_minibatch_kmeans: bool = False,
    n_init: int = 5,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: int = 42,
    min_cluster_size: int = 1,
    progress: Callable[[int, int, int], None] | None = None,
    check: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Scan K x covariance structure with the Fast-Funclu selection procedure.

    Each candidate keeps the highest-likelihood run over ``n_init`` starting
    seeds instead of aggregating seeds, then reports BIC, AIC, parameter
    counts, convergence and module sizes.  The selected model is the lowest
    BIC among candidates whose smallest module meets ``min_cluster_size``;
    ``delta_bic`` is measured against it.  S1-S3 share the SAD(1) covariance
    parameters across conditions; S4 is condition-specific.
    """
    from app.pure.fast_funclu_power import COVARIANCE_MODELS as ENGINE_MODELS
    from app.pure.fast_funclu_power import PowerFunClu

    if not data:
        raise ValueError("FunClu requires at least one condition")
    if k_min < 1:
        raise ValueError("k_min must be at least 1")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be at least 1")
    models = tuple(str(name).upper() for name in covariance_models)
    if not models or any(name not in ENGINE_MODELS for name in models):
        raise ValueError(
            "covariance_models must be a non-empty subset of "
            + ", ".join(ENGINE_MODELS)
        )
    frames, valid_columns = finite_condition_frames(data)
    n_features = len(valid_columns)
    effective_max = min(int(k_max), n_features)
    ks = list(range(int(k_min), effective_max + 1, int(step)))
    if not ks:
        raise ValueError(
            f"No K values to scan: k_min={k_min}, k_max={k_max}, features={n_features}"
        )
    n_init = max(1, int(n_init))

    rows: list[dict[str, Any]] = []
    for index, k in enumerate(ks):
        for model_name in models:
            best = None
            errors: list[str] = []
            for offset in range(n_init):
                if check is not None:
                    check()
                try:
                    model = PowerFunClu(
                        n_components=k,
                        max_iter=max_iter,
                        tol=tol,
                        random_state=int(random_state) + offset,
                        covariance_model=model_name,
                        init_params=init_params,
                        use_minibatch_kmeans=use_minibatch_kmeans,
                    )
                    model.fit(data, parameter_data=parameter_data)
                except Exception as exc:  # noqa: BLE001 - a single start may fail
                    errors.append(f"{model_name}: {exc}")
                    continue
                if best is None or model.log_likelihood > best.log_likelihood:
                    best = model
            if best is None:
                rows.append(
                    {
                        "K": k,
                        "covariance_model": model_name,
                        "BIC": float("nan"),
                        "AIC": float("nan"),
                        "init_params": init_params,
                        "use_minibatch_kmeans": bool(use_minibatch_kmeans),
                        "error": "; ".join(errors),
                    }
                )
                continue
            labels = np.asarray(best.labels, dtype=int)
            counts = np.bincount(labels, minlength=k)
            sizes = [int(count) for count in counts]
            n_parameters = int(best.n_params)
            n_covariance = int(best.n_covariance_parameters)
            rows.append(
                {
                    "K": k,
                    "covariance_model": model_name,
                    "n_parameters": n_parameters,
                    "n_covariance_parameters": n_covariance,
                    "bic_penalty": float(
                        n_parameters * math.log(max(int(best.n_features), 1))
                    ),
                    "BIC": float(best.bic),
                    "AIC": float(
                        -2.0 * float(best.log_likelihood) + 2.0 * n_parameters
                    ),
                    "log_likelihood": float(best.log_likelihood),
                    "SI": _silhouette_index(
                        data, list(best.common_cols), labels
                    ),
                    "converged": bool(best.converged),
                    "n_iter": int(best.n_iter_run),
                    "n_init": n_init,
                    "init_params": init_params,
                    "init_params_used": best.init_params_used,
                    "use_minibatch_kmeans": bool(use_minibatch_kmeans),
                    "n_features": int(best.n_features),
                    "n_conditions": int(best.n_conditions),
                    "min_cluster_size": int(min(sizes)),
                    "max_cluster_size": int(max(sizes)),
                    "cluster_sizes": ";".join(str(size) for size in sizes),
                    "error": "; ".join(errors),
                }
            )
        if progress is not None:
            progress(k, index + 1, len(ks))

    table = pd.DataFrame(rows)
    valid = table.dropna(subset=["BIC"])
    if valid.empty:
        raise ValueError("No BIC candidate could be fitted")
    table["eligible"] = (
        table["min_cluster_size"].fillna(-1).astype(int) >= int(min_cluster_size)
    )
    eligible = table[table["eligible"] & table["BIC"].notna()]
    pool = eligible if not eligible.empty else valid
    best_index = pool["BIC"].idxmin()
    table["delta_bic"] = table["BIC"] - table.loc[best_index, "BIC"]
    table["selected"] = False
    table.loc[best_index, "selected"] = True
    leading = [
        "K",
        "covariance_model",
        "BIC",
        "delta_bic",
        "selected",
        "eligible",
        "AIC",
        "n_parameters",
        "n_covariance_parameters",
        "bic_penalty",
        "log_likelihood",
        "SI",
        "converged",
        "min_cluster_size",
        "max_cluster_size",
        "cluster_sizes",
        "n_iter",
        "n_init",
        "init_params",
        "init_params_used",
        "use_minibatch_kmeans",
        "n_features",
        "n_conditions",
        "error",
    ]
    return table[[column for column in leading if column in table.columns]]


def save_bic_comparison(bic_df: pd.DataFrame, path: Path) -> Path:
    """BIC across K for every fitted SAD(1) covariance structure."""
    import matplotlib.pyplot as plt

    colors = {
        "S1": "#1f77b4",
        "S2": "#ff7f0e",
        "S3": "#2ca02c",
        "S4": "#9467bd",
    }
    figure, axes = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    models = [
        name
        for name in ("S1", "S2", "S3", "S4")
        if name in set(bic_df["covariance_model"])
    ]
    for name in models:
        subset = (
            bic_df[bic_df["covariance_model"] == name]
            .dropna(subset=["BIC"])
            .sort_values("K")
        )
        if subset.empty:
            continue
        axes.plot(
            subset["K"],
            subset["BIC"],
            marker="o",
            linewidth=1.8,
            color=colors[name],
            label=name,
        )
        local = subset.loc[subset["BIC"].idxmin()]
        axes.scatter(
            [local["K"]], [local["BIC"]], s=55, color=colors[name], zorder=5
        )
    selected = bic_df[bic_df["selected"]]
    if not selected.empty:
        row = selected.iloc[0]
        axes.scatter(
            [row["K"]],
            [row["BIC"]],
            marker="*",
            s=220,
            color="black",
            zorder=6,
            label=f"selected: {row['covariance_model']}, K={int(row['K'])}",
        )
    axes.set_xlabel("Number of clusters K")
    axes.set_ylabel("BIC (lower is better)")
    axes.set_title("FunClu BIC across K and SAD(1) covariance structures")
    axes.set_xticks(sorted(int(k) for k in bic_df["K"].dropna().unique()))
    axes.grid(alpha=0.25, linestyle=":")
    axes.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path
