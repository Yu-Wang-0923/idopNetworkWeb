"""Filesystem storage and archive safety helpers."""
from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from .config import Settings

_UNSAFE = re.compile(r"[^\w.\-]+", re.UNICODE)


class StorageError(Exception):
    """Base class for storage and upload errors."""


class UploadTooLarge(StorageError):
    pass


class UnsafeArchive(StorageError):
    pass


def safe_filename(name: str | None) -> str:
    base = Path((name or "file").replace("\\", "/")).name
    cleaned = _UNSAFE.sub("_", base).strip(" ._")
    return cleaned[:200] or "file"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_zip_member(name: str) -> None:
    if not name or "\x00" in name:
        raise UnsafeArchive("Archive contains an empty or invalid entry name")
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        raise UnsafeArchive("Archive contains an absolute path")
    if re.match(r"^[A-Za-z]:", normalized):
        raise UnsafeArchive("Archive contains a drive-qualified path")
    parts = PurePosixPath(normalized).parts
    if ".." in parts:
        raise UnsafeArchive("Archive contains a parent-directory reference")


def inspect_zip(path: Path, settings: Settings) -> list[str]:
    """Validate a ZIP and return its entry names without extracting it."""
    if not zipfile.is_zipfile(path):
        raise UnsafeArchive("File is not a valid ZIP archive")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > settings.max_zip_entries:
            raise UnsafeArchive("Archive contains too many entries")
        total_uncompressed = 0
        for info in infos:
            _check_zip_member(info.filename)
            total_uncompressed += info.file_size
            if info.compress_size > 0:
                ratio = info.file_size / info.compress_size
                if ratio > settings.max_zip_ratio:
                    raise UnsafeArchive("Archive compression ratio is suspiciously high")
        if total_uncompressed > settings.max_zip_uncompressed_bytes:
            raise UnsafeArchive("Archive expands beyond the size limit")
        return [info.filename for info in infos]


def detect_zip_kind(names: list[str]) -> str:
    normalized = [name.replace("\\", "/") for name in names]
    if any(name.endswith("from_to.csv") for name in normalized):
        return "netrecon_export"
    joined = "\n".join(normalized)
    if "cluster_centers/" in joined and "cluster_members/" in joined:
        return "funclu_k_export"
    if any(name.endswith("quasi_dynamic.csv") for name in normalized) and any(
        name.endswith("curve_sample.csv") for name in normalized
    ):
        return "curve_fitting_export"
    return "unknown"


def zip_conditions(names: list[str]) -> list[str]:
    conditions: set[str] = set()
    for name in names:
        normalized = name.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.name in {"quasi_dynamic.csv", "curve_sample.csv", "from_to.csv"}:
            parent = str(path.parent)
            if parent and parent != ".":
                conditions.add(parent)
        elif normalized.startswith("cluster_centers/"):
            parts = path.parts
            if len(parts) >= 3:
                conditions.add(parts[1])
    return sorted(conditions)


def zip_members(names: list[str]) -> list[str]:
    return sorted(
        name.replace("\\", "/")
        for name in names
        if name.replace("\\", "/").endswith("from_to.csv")
    )


def detect_kind(path: Path, settings: Settings) -> tuple[str, list[str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv", []
    if suffix == ".zip":
        names = inspect_zip(path, settings)
        return detect_zip_kind(names), names
    return "unknown", []


class Storage:
    def __init__(self, settings: Settings):
        self.settings = settings

    def resolve_within(self, root: Path, relative: str | Path) -> Path:
        relative_path = Path(relative)
        if relative_path.is_absolute():
            raise StorageError("Absolute paths are not allowed")
        if ".." in relative_path.parts:
            raise StorageError("Parent-directory references are not allowed")
        root_resolved = root.resolve()
        candidate = (root_resolved / relative_path).resolve()
        if not candidate.is_relative_to(root_resolved):
            raise StorageError("Path escapes its storage root")
        return candidate

    def dataset_dir(self, dataset_id: str) -> Path:
        return self.settings.uploads_dir / dataset_id

    def dataset_path(self, stored_path: str) -> Path:
        return self.resolve_within(self.settings.data_dir, stored_path)

    def run_dir(self, run_id: str) -> Path:
        return self.settings.runs_dir / run_id

    def run_work_dir(self, run_id: str) -> Path:
        return self.settings.work_dir / run_id

    def run_output_dir(self, run_id: str) -> Path:
        return self.run_work_dir(run_id) / "output"

    def run_artifacts_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "artifacts"

    def artifact_path(self, run_id: str, relative_path: str) -> Path:
        return self.resolve_within(self.run_artifacts_dir(run_id), relative_path)

    def save_upload(self, dataset_id: str, filename: str, fileobj, max_bytes: int):
        directory = self.dataset_dir(dataset_id)
        directory.mkdir(parents=True, exist_ok=True)
        safe_name = safe_filename(filename)
        destination = directory / safe_name
        temporary = directory / (safe_name + ".part")
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("wb") as handle:
                while True:
                    chunk = fileobj.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise UploadTooLarge("Upload exceeds the size limit")
                    digest.update(chunk)
                    handle.write(chunk)
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            shutil.rmtree(directory, ignore_errors=True)
            raise
        relative = str(destination.relative_to(self.settings.data_dir)).replace(os.sep, "/")
        mime = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
        return relative, size, digest.hexdigest(), safe_name, mime
