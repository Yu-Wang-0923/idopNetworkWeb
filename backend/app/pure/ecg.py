"""Pure ECG fitting figure helpers (no Streamlit)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from idopnetwork.curve_fitting import ecg_plot


def _with_time_column(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "time" not in result.columns and str(result.index.name).lower() == "time":
        result.insert(0, "time", result.index.to_numpy(dtype=float))
    return result


def render_ecg_fitting(
    observed: pd.DataFrame,
    fitted: pd.DataFrame,
    output_dir: Path,
    *,
    sampling_rate: float = 100.0,
    fit_label: str = "SWT fit",
) -> tuple[list[str], list[Path]]:
    """Render one full/zoom figure per lead and return (leads, png paths)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    leads = [column for column in observed.columns if column != "time"]
    paths = ecg_plot.plot_fitting(
        _with_time_column(observed),
        _with_time_column(fitted),
        output_dir,
        sampling_rate=sampling_rate,
        fit_label=fit_label,
        formats=("png", "pdf", "svg"),
    )
    if len(paths) != len(leads):
        raise RuntimeError("ECG renderer returned an unexpected number of figures")
    return leads, paths
