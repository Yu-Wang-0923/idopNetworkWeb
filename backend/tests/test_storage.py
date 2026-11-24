from __future__ import annotations

import io
import zipfile

from app.config import get_settings
from app.storage import (
    Storage,
    StorageError,
    UnsafeArchive,
    detect_zip_kind,
    inspect_zip,
    safe_filename,
)


def test_safe_filename_strips_paths():
    assert safe_filename("..\\evil name.csv") == "evil_name.csv"
    assert safe_filename("教学 示例.csv") == "教学_示例.csv"
    assert safe_filename("") == "file"


def test_detect_zip_kinds():
    assert (
        detect_zip_kind(["cond/quasi_dynamic.csv", "cond/curve_sample.csv"])
        == "curve_fitting_export"
    )
    assert (
        detect_zip_kind(
            [
                "labels.csv",
                "cluster_centers/cond/cluster_center_curve_sample.csv",
                "cluster_members/cond/M1_curve_sample.csv",
            ]
        )
        == "funclu_k_export"
    )
    assert detect_zip_kind(["inter_cluster/cond/from_to.csv"]) == "netrecon_export"
    assert detect_zip_kind(["random.txt"]) == "unknown"


def test_inspect_zip_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../evil.csv", "a,b\n1,2\n")
    try:
        inspect_zip(archive_path, get_settings())
    except UnsafeArchive:
        pass
    else:
        raise AssertionError("path traversal archive was accepted")


def test_inspect_zip_rejects_suspicious_ratio(tmp_path):
    archive_path = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("payload.csv", "0" * (2 * 1024 * 1024))
    try:
        inspect_zip(archive_path, get_settings())
    except UnsafeArchive:
        pass
    else:
        raise AssertionError("high-ratio archive was accepted")


def test_storage_resolve_within(tmp_path):
    storage = Storage(get_settings())
    root = tmp_path / "root"
    root.mkdir()
    assert storage.resolve_within(root, "a/b.txt") == (root / "a" / "b.txt").resolve()
    for bad in ("../x", "/abs", "a/../../x"):
        try:
            storage.resolve_within(root, bad)
        except StorageError:
            continue
        raise AssertionError(f"path was accepted: {bad}")
