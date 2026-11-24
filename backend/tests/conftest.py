"""Test configuration: isolate runtime data before importing the app."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

DATA_DIR = Path(tempfile.mkdtemp(prefix="idop-network-web-tests-"))
os.environ["IDOP_DATA_DIR"] = str(DATA_DIR)
os.environ["IDOP_SECRET_KEY"] = "test-secret-key"
os.environ["IDOP_RATE_LIMIT_PER_MINUTE"] = "100000"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db import init_db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _database():
    init_db()
    yield


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def other_client():
    with TestClient(app) as test_client:
        yield test_client
