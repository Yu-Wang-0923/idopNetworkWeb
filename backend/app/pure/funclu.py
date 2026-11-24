# Copied from https://github.com/Yu-Wang-0923/idopNetworkAPP@fc5dc2f5a903c7fdbdadeef903771f9852f05b8c
# packages/idopnetwork-app/src/idopnetwork_app/pages/2_FunClu.py:112-227 and :63-109
# The previous Streamlit app is frozen; this copy is the maintained source for idopNetwork Web.
"""Streamlit-free FunClu helpers: curve ZIP loading, K export, BIC summaries."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

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


def checkmark_candidates(
    bic_df: pd.DataFrame, *, max_short_arm: int = 3
) -> list[int]:
    """K values that sit at a checkmark-like local BIC minimum."""
    if bic_df is None or "K" not in bic_df.columns or "BIC" not in bic_df.columns:
        return []
    frame = bic_df.dropna(subset=["K", "BIC"]).sort_values("K")
    if len(frame) < 3:
        return []
    ks = frame["K"].to_numpy(dtype=int)
    bics = frame["BIC"].to_numpy(dtype=float)
    scale = float(np.nanmax(np.abs(bics))) if bics.size else 1.0
    eps = max(1e-9, scale * 1e-12)

    candidates: list[int] = []
    for idx in range(1, len(bics) - 1):
        if not (bics[idx] + eps < bics[idx - 1] and bics[idx] + eps < bics[idx + 1]):
            continue
        left_len = 0
        last = bics[idx]
        for j in range(idx - 1, -1, -1):
            if bics[j] > last + eps:
                left_len += 1
                last = bics[j]
            else:
                break
        right_len = 0
        last = bics[idx]
        for j in range(idx + 1, len(bics)):
            if bics[j] > last + eps:
                right_len += 1
                last = bics[j]
            else:
                break
        short_arm = min(left_len, right_len)
        long_arm = max(left_len, right_len)
        if (
            left_len >= 1
            and right_len >= 1
            and short_arm <= max_short_arm
            and long_arm > short_arm
        ):
            candidates.append(int(ks[idx]))
    return candidates


def recommend_best_k(
    bic_df: pd.DataFrame, *, mode: str
) -> tuple[int | None, int | None, list[int]]:
    """Return (recommended, raw_min_bic, candidates) for the BIC table."""
    valid = bic_df.dropna(subset=["BIC"])
    if valid.empty:
        return None, None, []
    raw_best = int(valid.loc[valid["BIC"].idxmin(), "K"])
    if mode == "default":
        candidates = checkmark_candidates(bic_df)
        return (min(candidates) if candidates else None), raw_best, candidates
    if "passes_min_cluster_size" in valid.columns:
        eligible = valid[valid["passes_min_cluster_size"].astype(bool)]
    else:
        eligible = valid
    pool = eligible if not eligible.empty else valid
    best = int(pool.loc[pool["BIC"].idxmin(), "K"])
    return best, raw_best, []


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
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return path
