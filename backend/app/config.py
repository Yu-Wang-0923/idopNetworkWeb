"""Runtime configuration for the idopNetwork Web backend."""
from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="IDOP_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = REPO_ROOT / "data"
    secret_key: str = ""

    session_cookie: str = "idop_workspace"
    session_max_age_days: int = 180
    cookie_secure: bool = False

    allowed_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:8000,http://127.0.0.1:8000"
    )

    max_csv_bytes: int = 100 * 1024 * 1024
    max_zip_bytes: int = 500 * 1024 * 1024
    max_zip_entries: int = 5000
    max_zip_uncompressed_bytes: int = 2 * 1024 * 1024 * 1024
    max_zip_ratio: int = 200

    retention_days: int = 30
    rate_limit_per_minute: int = 60
    max_concurrent_runs_per_workspace: int = 5
    max_runs_per_day_per_workspace: int = 100
    max_storage_bytes_per_workspace: int = 2 * 1024 * 1024 * 1024
    disabled_tasks: str = ""
    beta_tasks: str = ""

    worker_poll_seconds: float = 1.0
    worker_heartbeat_seconds: float = 10.0
    worker_stale_seconds: float = 120.0
    cancel_grace_seconds: float = 15.0

    @property
    def allowed_origins_list(self) -> list[str]:
        return [item.strip() for item in self.allowed_origins.split(",") if item.strip()]

    @property
    def disabled_task_keys(self) -> set[str]:
        return {item.strip() for item in self.disabled_tasks.split(",") if item.strip()}

    @property
    def beta_task_keys(self) -> set[str]:
        return {item.strip() for item in self.beta_tasks.split(",") if item.strip()}

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.uploads_dir, self.runs_dir, self.work_dir, self.tmp_dir):
            path.mkdir(parents=True, exist_ok=True)

    def resolved_secret_key(self) -> str:
        if self.secret_key:
            return self.secret_key
        self.ensure_dirs()
        key_file = self.data_dir / "secret.key"
        if key_file.exists():
            value = key_file.read_text(encoding="utf-8").strip()
            if value:
                return value
        value = secrets.token_urlsafe(48)
        key_file.write_text(value, encoding="utf-8")
        return value


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


def clear_settings_cache() -> None:
    get_settings.cache_clear()
