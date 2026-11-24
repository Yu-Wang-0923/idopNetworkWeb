"""Inspect uploaded CSVs and ZIP archives without extracting them."""
from __future__ import annotations

from pathlib import Path

from .config import Settings
from .storage import detect_kind, zip_conditions, zip_members


def inspect_upload(path: Path, original_filename: str, settings: Settings) -> dict:
    kind, names = detect_kind(path, settings)
    if kind == "csv":
        return {"kind": "csv", "conditions": [original_filename], "members": []}
    return {
        "kind": kind,
        "conditions": zip_conditions(names),
        "members": zip_members(names),
    }
