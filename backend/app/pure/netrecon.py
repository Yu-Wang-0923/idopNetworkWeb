# Adapted from https://github.com/Yu-Wang-0923/idopNetworkAPP@fc5dc2f5a903c7fdbdadeef903771f9852f05b8c
# packages/idopnetwork-app/src/idopnetwork_app/pages/3_NetRecon.py:45-101, 200-216,
# 311-498 and :114-138 (MIT License).
# The previous Streamlit app is frozen; this copy is the maintained source for
# idopNetwork Web.  Fitting itself still comes from the pinned idopnetwork library.
"""Single-layer NetRecon helpers: curve loading, fitting, network HTML, export."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from idopnetwork.curve_fitting.fitting import get_power_function_params
from idopnetwork.network.construction import (
    IDOPRegressor,
    align_response_to_design,
)
from idopnetwork.network.dynamic_idop import (
    DynamicIDOPRegressor,
    observed_for_plot,
)
from idopnetwork.network.static_idop import StaticIDOPRegressor

IDOP_ALGORITHM_ASGL = "asgl"
IDOP_ALGORITHM_STATIC = "static_idop"
IDOP_ALGORITHM_DYNAMIC = "dynamic_idop"
IDOP_ALGORITHMS = (
    IDOP_ALGORITHM_ASGL,
    IDOP_ALGORITHM_STATIC,
    IDOP_ALGORITHM_DYNAMIC,
)

DYNAMIC_DEFAULT_FS = 100.0
DYNAMIC_DEFAULT_WINDOW = 0
DYNAMIC_DEFAULT_N_FOURIER = 0

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
VIS_NETWORK = STATIC_DIR / "vis-network.min.js"

# Figure defaults tuned for the all-edge view: the highest out-degree nodes
# sit in the inner hub ring and labels/edges stay legible on a full network.
NETWORK_CENTER_HUBS = 5
NETWORK_NODE_SCALE = 2.0
NETWORK_LABEL_SCALE = 2.0
NETWORK_EDGE_SCALE = 2.2
NETWORK_EDGE_OPACITY = 0.9


def _center_hub_count(adj_df: pd.DataFrame) -> int:
    """High out-degree nodes only; at most NETWORK_CENTER_HUBS.

    Self loops are excluded: the layout's out-degree counts cross edges only.
    """
    values = adj_df.abs().to_numpy(dtype=float, copy=True)
    np.fill_diagonal(values, 0.0)
    out_degree = (values > 1e-12).sum(axis=1)
    strong = int((out_degree >= 2).sum())
    return max(1, min(NETWORK_CENTER_HUBS, strong or 1))
_VIS_CDN = (
    '<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/'
    'vis-network.min.js"></script>'
)

# The artifact iframe is sandboxed without allow-same-origin, so plain
# localStorage access throws and would stop the script before its initial
# network.fit().  These wrappers keep the page working (layouts just are not
# persisted in that context).
_SAFE_STORAGE = """
<script>
function _idopSafeStorageGet(key) {
  try { return window.localStorage.getItem(key); } catch (error) { return null; }
}
function _idopSafeStorageSet(key, value) {
  try { window.localStorage.setItem(key, value); } catch (error) {}
}
</script>
"""


def load_curve_maps(
    zip_bytes: bytes,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Read quasi_dynamic / curve_sample tables from a Curve Fitting ZIP."""
    quasi: dict[str, pd.DataFrame] = {}
    curve: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for name in archive.namelist():
            normalized = name.replace("\\", "/")
            if normalized.endswith("quasi_dynamic.csv"):
                condition = normalized.rsplit("/", 1)[0] if "/" in normalized else normalized
                with archive.open(name) as handle:
                    quasi[condition] = pd.read_csv(handle, index_col=0)
            elif normalized.endswith("curve_sample.csv"):
                condition = normalized.rsplit("/", 1)[0] if "/" in normalized else normalized
                with archive.open(name) as handle:
                    curve[condition] = pd.read_csv(handle, index_col=0)
    return (
        dict(sorted(quasi.items())),
        dict(sorted(curve.items())),
    )


def finite_network_frames(
    curve_sample: pd.DataFrame, quasi_dynamic: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Keep shared features whose values are finite in both tables."""
    common = [
        str(column)
        for column in curve_sample.columns
        if column in quasi_dynamic.columns
    ]
    valid = [
        column
        for column in common
        if np.isfinite(curve_sample[column].to_numpy(dtype=float)).all()
        and np.isfinite(quasi_dynamic[column].to_numpy(dtype=float)).all()
    ]
    if len(valid) < 2:
        raise ValueError("NetRecon needs at least 2 shared finite features")
    return curve_sample.loc[:, valid], quasi_dynamic.loc[:, valid], valid


def make_idop_model(
    *,
    algorithm: str,
    max_order: int,
    nonneg_self: bool,
    max_interactions: int,
    lasso_alpha: float = 0.1,
    lasso_windows: int = 1,
    lasso_threshold: float = 0.5,
    n_fourier: int = DYNAMIC_DEFAULT_N_FOURIER,
    dynamic_window: int = DYNAMIC_DEFAULT_WINDOW,
    dynamic_fs: float = DYNAMIC_DEFAULT_FS,
):
    """Construct the selected NetRecon model (ASGL / static / dynamic)."""
    if str(algorithm) == IDOP_ALGORITHM_DYNAMIC:
        return DynamicIDOPRegressor(
            n_fourier=None if int(n_fourier) == 0 else int(n_fourier),
            r_legendre=int(max_order),
            window=None if int(dynamic_window) == 0 else int(dynamic_window),
            fs=float(dynamic_fs),
            lasso_alpha=float(lasso_alpha),
            lasso_windows=None if int(lasso_windows) == 0 else int(lasso_windows),
            lasso_threshold=float(lasso_threshold),
            mix=0.5,
            nonneg_self=bool(nonneg_self),
            max_interactions=int(max_interactions),
        )
    if str(algorithm) == IDOP_ALGORITHM_STATIC:
        return StaticIDOPRegressor(
            max_order=int(max_order),
            mix=0.5,
            fix_mix=False,
            nonneg_self=bool(nonneg_self),
            max_interactions=int(max_interactions),
            adaptive_weights=False,
            lasso_alpha=float(lasso_alpha),
            lasso_windows=max(1, int(lasso_windows)),
            lasso_threshold=float(lasso_threshold),
        )
    return IDOPRegressor(
        max_order=int(max_order),
        mix=0.5,
        fix_mix=False,
        nonneg_self=bool(nonneg_self),
        max_interactions=int(max_interactions),
        adaptive_weights=False,
    )


def fit_single_network(
    curve_sample: pd.DataFrame,
    quasi_dynamic: pd.DataFrame,
    *,
    algorithm: str,
    max_order: int,
    nonneg_self: bool,
    max_interactions: int,
    adjacency_aggregation: str,
    lasso_alpha: float = 0.1,
    lasso_windows: int = 1,
    lasso_threshold: float = 0.5,
    n_fourier: int = DYNAMIC_DEFAULT_N_FOURIER,
    dynamic_window: int = DYNAMIC_DEFAULT_WINDOW,
    dynamic_fs: float = DYNAMIC_DEFAULT_FS,
) -> dict:
    """Fit one condition and return the single-layer result bundle."""
    if curve_sample.shape[1] < 2:
        raise ValueError("At least two curves are required to build a network")
    response = quasi_dynamic.loc[:, list(curve_sample.columns)]
    model = make_idop_model(
        algorithm=algorithm,
        max_order=max_order,
        nonneg_self=nonneg_self,
        max_interactions=max_interactions,
        lasso_alpha=lasso_alpha,
        lasso_windows=lasso_windows,
        lasso_threshold=lasso_threshold,
        n_fourier=n_fourier,
        dynamic_window=dynamic_window,
        dynamic_fs=dynamic_fs,
    )
    power_parameters = (
        pd.DataFrame()
        if algorithm == IDOP_ALGORITHM_DYNAMIC
        else get_power_function_params(response)
    )
    model.fit(curve_sample, response, power_function_params=power_parameters)
    predicted = model.predict(curve_sample)
    effects = model.effect(curve_sample)
    adjacency = model.adjacency_matrix(
        curve_sample, aggregation=str(adjacency_aggregation)
    )
    design = model._design(curve_sample)
    observed = observed_for_plot(model, response)
    response_aligned = align_response_to_design(observed, design.index)
    return {
        "model": model,
        "quasi_dynamic_df": observed,
        "curve_sample_df": curve_sample,
        "design_X": design,
        "response_Y": response_aligned,
        "predicted_df": predicted,
        "effect_df_list": effects,
        "adj_df": adjacency,
        "adjacency_aggregation": str(adjacency_aggregation),
        "power_function_params": power_parameters,
    }


def adjacency_to_from_to(
    adj_df: pd.DataFrame, *, eps: float = 1e-12
) -> pd.DataFrame:
    """Convert adjacency to a from/to edge table, keeping self loops."""
    rows: list[dict[str, str | float]] = []
    for from_node in adj_df.index:
        for to_node in adj_df.columns:
            weight = float(adj_df.loc[from_node, to_node])
            if abs(weight) < eps:
                continue
            rows.append(
                {
                    "from": str(from_node),
                    "to": str(to_node),
                    "weight": weight,
                    "type": "+" if weight > 0 else "-",
                }
            )
    if not rows:
        return pd.DataFrame(columns=["from", "to", "weight", "type"])
    return pd.DataFrame(rows, columns=["from", "to", "weight", "type"])


def params_to_df(params: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"param": key, "value": value} for key, value in params.items()],
        columns=["param", "value"],
    )


def _to_csv_utf8_sig(df: pd.DataFrame, *, index: bool) -> bytes:
    return df.to_csv(index=index).encode("utf-8-sig")


def collect_network_export_artifacts(
    network: dict,
    *,
    extra_params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Extract adjacency, effects, edges, params and dynamic tables."""
    adj_df: pd.DataFrame = network["adj_df"]
    effect_df_list: list[pd.DataFrame] = network["effect_df_list"]
    model = network["model"]
    effect_map = {
        str(target): effect
        for target, effect in zip(adj_df.index, effect_df_list)
    }
    from_to_df = adjacency_to_from_to(adj_df)
    params: dict[str, object] = {
        "max_order": int(model.max_order),
        "alpha": float(model.alpha),
        "mix": float(model.mix),
        "nonneg_self": bool(model.nonneg_self),
        "max_interactions": int(model.max_interactions),
        "basis_type": model.basis_type,
        "ebic_gamma": float(model.ebic_gamma),
        "enforce_effect_constraints": bool(model.enforce_effect_constraints),
        "mse": float(model.mse_) if model.mse_ is not None else np.nan,
        "n_nodes": int(adj_df.shape[0]),
        "n_edges_nonzero": int(from_to_df.shape[0]),
    }
    if hasattr(model, "settings_"):
        params.update(model.settings_)
        params.update(
            {
                "solver_version": model.solver_version_,
                "edge_weight": "sign * energy_share",
                "alpha_ratio": model.lasso_alpha,
                "support_window": model.window,
                "support_stride": model.lasso_windows,
                "support_threshold": model.lasso_threshold,
            }
        )
    if extra_params:
        params.update(extra_params)
    return {
        "adj_df": adj_df,
        "effect_map": effect_map,
        "from_to_df": from_to_df,
        "params_df": params_to_df(params),
        "dynamic_tables": getattr(model, "ecg_exports_", {}),
    }


def build_singlelayer_export_zip(result: dict) -> bytes:
    """Build the single_layer_idopnetwork_export.zip contract."""
    artifacts = collect_network_export_artifacts(
        result,
        extra_params={
            "condition": result.get("condition", ""),
            "adjacency_aggregation": result.get("adjacency_aggregation", "mean"),
        },
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "adjacency_matrix.csv",
            _to_csv_utf8_sig(artifacts["adj_df"], index=True),
        )
        archive.writestr(
            "from_to.csv",
            _to_csv_utf8_sig(artifacts["from_to_df"], index=False),
        )
        archive.writestr(
            "params.csv",
            _to_csv_utf8_sig(artifacts["params_df"], index=False),
        )
        for name, table in artifacts["dynamic_tables"].items():
            archive.writestr(
                f"dynamic/{name}.csv", _to_csv_utf8_sig(table, index=False)
            )
        for target_name, effect_df in artifacts["effect_map"].items():
            archive.writestr(
                f"effect/{target_name}.csv",
                _to_csv_utf8_sig(effect_df, index=True),
            )
    return buffer.getvalue()


def build_effect_zip(effect_map: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for target_name, effect_df in effect_map.items():
            archive.writestr(
                f"effect/{target_name}.csv",
                _to_csv_utf8_sig(effect_df, index=True),
            )
    return buffer.getvalue()


def build_network_html(
    adj_df: pd.DataFrame, output_dir: Path, *, top_edges: int = 0
) -> Path:
    """Render the interactive hub-ring network page with offline vis-network.

    ``top_edges=0`` keeps every non-zero cross edge.  Only nodes that occur in
    the displayed edges enter the ring layout, so a positive ``top_edges``
    limit never leaves isolated nodes in the figure.
    """
    from idopnetwork.network import interactive as interactive_module
    from idopnetwork.network.interactive import (
        network_edges_from_adjacency,
        plot_network,
    )

    interactive_module.N_CENTER_HUBS = _center_hub_count(adj_df)
    interactive_module.DEFAULT_NODE_SCALE = NETWORK_NODE_SCALE
    interactive_module.DEFAULT_LABEL_SCALE = NETWORK_LABEL_SCALE
    interactive_module.DEFAULT_EDGE_SCALE = NETWORK_EDGE_SCALE
    interactive_module.DEFAULT_EDGE_OPACITY = NETWORK_EDGE_OPACITY

    edges = network_edges_from_adjacency(
        adj_df,
        target_node="",
        top_edges=None if int(top_edges) <= 0 else int(top_edges),
    )
    if edges.empty:
        raise ValueError("No non-zero edge is available for the network figure")
    modules = sorted(
        {str(value) for value in edges["source"]}
        | {str(value) for value in edges["target"]}
    )
    path = output_dir / "network.html"
    plot_network(
        edges,
        str(path),
        title="Single-layer IdopNetwork",
        group_name="network",
        group_dir=str(output_dir),
        modules=modules,
    )
    html = path.read_text(encoding="utf-8")
    if _VIS_CDN in html:
        script = VIS_NETWORK.read_text(encoding="utf-8")
        html = html.replace(_VIS_CDN, f"<script>{script}</script>")
    if "localStorage.getItem(" in html or "localStorage.setItem(" in html:
        html = html.replace("localStorage.getItem(", "_idopSafeStorageGet(")
        html = html.replace("localStorage.setItem(", "_idopSafeStorageSet(")
        html = html.replace("</head>", _SAFE_STORAGE + "</head>", 1)
    path.write_text(html, encoding="utf-8")
    return path
