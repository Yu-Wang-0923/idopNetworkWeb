"""Inspect uploaded CSVs and ZIP archives without extracting them."""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from .config import Settings
from .storage import detect_kind, zip_conditions, zip_members


def curve_feature_count(path: Path) -> int | None:
    """Column count (minus the index column) of the first curve_sample.csv."""
    try:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.replace("\\", "/").endswith("curve_sample.csv"):
                    with archive.open(name) as handle:
                        header = handle.readline().decode("utf-8", "replace").strip()
                    return max(0, len(header.split(",")) - 1)
    except (OSError, zipfile.BadZipFile):
        return None
    return None


def funclu_clusters(path: Path) -> list[str]:
    """Cluster names reported by a FunClu-K export (M1, M2, ...)."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            for member in ("cluster_sizes.csv", "labels.csv"):
                if member not in names:
                    continue
                with archive.open(member) as handle:
                    frame = pd.read_csv(handle)
                if "cluster" in frame.columns:
                    return sorted({str(value) for value in frame["cluster"]})
    except (OSError, zipfile.BadZipFile, ValueError):
        return []
    return []


def inspect_upload(path: Path, original_filename: str, settings: Settings) -> dict:
    kind, names = detect_kind(path, settings)
    if kind == "csv":
        return {
            "kind": "csv",
            "conditions": [original_filename],
            "members": [],
            "clusters": [],
        }
    return {
        "kind": kind,
        "conditions": zip_conditions(names),
        "members": zip_members(names),
        "features": curve_feature_count(path)
        if kind == "curve_fitting_export"
        else None,
        "clusters": funclu_clusters(path) if kind == "funclu_k_export" else [],
    }
