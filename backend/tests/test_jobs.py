from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from app.job_runner import execute_run

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def create_demo_run(client, **params):
    payload = {
        "task_key": "demo.echo",
        "params": {"message": "hello", "steps": 2, "delay_ms": 1, **params},
        "inputs": [],
    }
    response = client.post("/api/runs", json=payload)
    assert response.status_code == 202, response.text
    return response.json()


def test_task_registry_lists_demo(client):
    response = client.get("/api/tasks")
    assert response.status_code == 200
    tasks = {item["key"]: item for item in response.json()}
    assert "demo.echo" in tasks
    demo = tasks["demo.echo"]
    assert demo["enabled"] is True
    assert demo["params_schema"]["type"] == "object"
    assert [stage["key"] for stage in demo["stages"]][0] == "demo.stage.prepare"


def test_demo_run_succeeds_and_publishes_artifacts(client):
    run = create_demo_run(client)
    assert run["status"] == "queued"
    assert run["progress"] == 0

    assert execute_run(run["id"]) == 0

    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["status"] == "succeeded"
    assert detail["progress"] == 100
    assert detail["message"]["key"] == "run.message.succeeded"
    names = {artifact["logical_name"] for artifact in detail["artifacts"]}
    assert names == {"result.txt", "summary.json"}
    primary = next(a for a in detail["artifacts"] if a["is_primary"])
    assert primary["logical_name"] == "result.txt"

    download = client.get(f"/api/artifacts/{primary['id']}/download")
    assert download.status_code == 200
    assert "hello" in download.text

    logs = client.get(f"/api/runs/{run['id']}/logs").json()
    assert len(logs) >= 3
    keys = [entry["message_key"] for entry in logs if entry["message_key"]]
    assert "demo.message.step" in keys


def test_job_runner_child_process(client):
    run = create_demo_run(client)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_ROOT)
    result = subprocess.run(
        [sys.executable, "-m", "app.job_runner", run["id"]],
        cwd=BACKEND_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["status"] == "succeeded"


def test_cancel_queued_run(client):
    run = create_demo_run(client)
    response = client.post(f"/api/runs/{run['id']}/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert execute_run(run["id"]) == 0
    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["status"] == "cancelled"


def test_retry_creates_new_run(client):
    run = create_demo_run(client)
    assert execute_run(run["id"]) == 0
    response = client.post(f"/api/runs/{run['id']}/retry")
    assert response.status_code == 202
    retried = response.json()
    assert retried["id"] != run["id"]
    assert retried["status"] == "queued"
    assert retried["params"] == run["params"]


def test_invalid_params_are_rejected(client):
    response = client.post(
        "/api/runs",
        json={"task_key": "demo.echo", "params": {"steps": 99}, "inputs": []},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "run.invalid_params"


def test_unknown_task_is_rejected(client):
    response = client.post(
        "/api/runs", json={"task_key": "nope", "params": {}, "inputs": []}
    )
    assert response.status_code == 404


def test_workspace_isolation_and_share_links(client, other_client):
    run = create_demo_run(client)
    assert execute_run(run["id"]) == 0

    own_list = client.get("/api/runs").json()
    assert any(item["id"] == run["id"] for item in own_list)

    other_list = other_client.get("/api/runs").json()
    assert all(item["id"] != run["id"] for item in other_list)

    shared = other_client.get(f"/api/runs/{run['id']}")
    assert shared.status_code == 200
    assert shared.json()["id"] == run["id"]

    forbidden = other_client.post(f"/api/runs/{run['id']}/cancel")
    assert forbidden.status_code == 403
