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


def build_funclu_k_export_zip(
    *,
    model: Any,
    cond_names: list[str],
    curve_sample_dict: dict[str, pd.DataFrame],
    quasi_dynamic_dict: dict[str, pd.DataFrame],
) -> bytes:
    """Build the FunClu-K ZIP consumed by multi-layer NetRecon."""
    if model.labels is None or model.common_cols is None:
        raise ValueError("FunClu-K result is incomplete: missing labels/common columns.")

    labels_np = model.labels.detach().cpu().numpy().astype(int)
    common_cols = list(model.common_cols)
    if len(labels_np) != len(common_cols):
        raise ValueError("FunClu-K labels do not match common curve_sample columns.")
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

            center_times, center_curves = model.get_cluster_curves(cond_idx)
            center_df = pd.DataFrame(
                center_curves.T,
                index=center_times,
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
    parameter_data: list[pd.DataFrame],
    *,
    k_min: int = 2,
    k_max: int = 10,
    step: int = 1,
    n_init: int = 5,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: int = 42,
    min_cluster_size: int = 1,
    progress: Callable[[int, int, int], None] | None = None,
    check: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Scan K with the Fast-Funclu selection procedure.

    Each candidate keeps the highest-likelihood run over ``n_init`` starting
    seeds instead of aggregating seeds, then reports BIC, AIC, parameter
    counts, convergence and module sizes.  The selected model is the lowest
    BIC among candidates whose smallest module meets ``min_cluster_size``;
    ``delta_bic`` is measured against it.  The core model exposes S4
    (condition-specific phi and gamma, shared across modules) only.
    """
    from idopnetwork.clustering.funclu import FunClu

    if not data:
        raise ValueError("FunClu requires at least one condition")
    if k_min < 1:
        raise ValueError("k_min must be at least 1")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be at least 1")
    n_features = min(int(frame.shape[1]) for frame in data)
    effective_max = min(int(k_max), n_features)
    ks = list(range(int(k_min), effective_max + 1, int(step)))
    if not ks:
        raise ValueError(
            f"No K values to scan: k_min={k_min}, k_max={k_max}, features={n_features}"
        )
    n_init = max(1, int(n_init))

    rows: list[dict[str, Any]] = []
    for index, k in enumerate(ks):
        best = None
        errors: list[str] = []
        for offset in range(n_init):
            if check is not None:
                check()
            try:
                model = FunClu(
                    n_components=k,
                    max_iter=max_iter,
                    tol=tol,
                    random_state=int(random_state) + offset,
                )
                model.fit(data, parameter_data=parameter_data)
            except Exception as exc:  # noqa: BLE001 - a single start may fail
                errors.append(str(exc))
                continue
            if best is None or model.log_likelihood > best.log_likelihood:
                best = model
        if best is None:
            rows.append(
                {
                    "K": k,
                    "covariance_model": "S4",
                    "BIC": float("nan"),
                    "AIC": float("nan"),
                    "error": "; ".join(errors),
                }
            )
        else:
            labels = best.labels.detach().cpu().numpy().astype(int)
            counts = np.bincount(labels, minlength=k)
            sizes = [int(count) for count in counts]
            n_parameters = int(best.n_params)
            rows.append(
                {
                    "K": k,
                    "covariance_model": "S4",
                    "n_parameters": n_parameters,
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
        "bic_penalty",
        "log_likelihood",
        "SI",
        "converged",
        "min_cluster_size",
        "max_cluster_size",
        "cluster_sizes",
        "n_iter",
        "n_init",
        "n_features",
        "n_conditions",
        "error",
    ]
    return table[[column for column in leading if column in table.columns]]


def save_bic_elbow(
    bic_df: pd.DataFrame,
    path: Path,
    *,
    best_k: int | None,
    candidate_ks: list[int] | None = None,
    auto_select_best: bool = True,
) -> Path:
    """Render the BIC elbow figure with the upstream matplotlib style."""
    from idopnetwork.clustering.plot import plot_bic_elbow

    figure = plot_bic_elbow(
        bic_results=bic_df,
        best_K=best_k,
        candidate_Ks=candidate_ks,
        auto_select_best=auto_select_best,
        show_in_streamlit=False,
        title="FunClu BIC across K (S4 covariance)",
        ylabel="BIC (lower is better)",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return path
