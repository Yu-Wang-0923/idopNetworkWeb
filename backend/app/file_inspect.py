"""Inspect uploaded CSVs and ZIP archives without extracting them."""
from __future__ import annotations

import zipfile
from pathlib import Path

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


def inspect_upload(path: Path, original_filename: str, settings: Settings) -> dict:
    kind, names = detect_kind(path, settings)
    if kind == "csv":
        return {"kind": "csv", "conditions": [original_filename], "members": []}
    return {
        "kind": kind,
        "conditions": zip_conditions(names),
        "members": zip_members(names),
        "features": curve_feature_count(path)
        if kind == "curve_fitting_export"
        else None,
    }
