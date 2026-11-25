from __future__ import annotations

import io
import json
import uuid
import zipfile

import pandas as pd
from sqlalchemy import select

from app.config import Settings, get_settings
from app.db import get_session_factory
from app.main import app
from app.models import Artifact, Run, Workspace, utcnow
from app.storage import Storage, sha256_file


def _zip_bytes(prefix: str = "") -> bytes:
    adjacency = pd.DataFrame(
        [[0.5, 1.0, 0.3], [0.0, 0.2, -0.6], [0.0, 0.0, 0.4]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    effects = {
        "A": pd.DataFrame(
            {"A": [0.1, 0.2, 0.3], "B": [0.0] * 3, "C": [0.0] * 3},
            index=[0.0, 1.0, 2.0],
        ),
        "B": pd.DataFrame(
            {"A": [0.0] * 3, "B": [0.2, 0.1, 0.0], "C": [0.0] * 3},
            index=[0.0, 1.0, 2.0],
        ),
        "C": pd.DataFrame(
            {"A": [0.0] * 3, "B": [0.0] * 3, "C": [0.3, 0.2, 0.1]},
            index=[0.0, 1.0, 2.0],
        ),
    }
    intercepts = pd.DataFrame(
        {"target": ["A", "B", "C"], "intercept": [0.5, 0.2, 0.1]}
    )
    base = f"{prefix}/" if prefix else ""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{base}adjacency_matrix.csv", adjacency.to_csv())
        archive.writestr(
            f"{base}from_to.csv", "from,to,weight\nA,B,1.0\nB,C,-0.6\nA,C,0.3\n"
        )
        archive.writestr(f"{base}intercepts.csv", intercepts.to_csv(index=False))
        for target, frame in effects.items():
            archive.writestr(f"{base}effect/{target}.csv", frame.to_csv())
    return buffer.getvalue()


def _workspace_id(client) -> str:
    client.get("/api/session")
    with get_session_factory()() as db:
        workspace = db.scalars(
            select(Workspace).order_by(Workspace.last_seen_at.desc())
        ).first()
        assert workspace is not None
        return workspace.id


def _seed_run(
    workspace_id: str,
    task_key: str,
    content: bytes,
    *,
    logical_name: str,
    semantic_type: str,
) -> str:
    settings = get_settings()
    storage = Storage(settings)
    run_id = str(uuid.uuid4())
    artifacts_dir = storage.run_artifacts_dir(run_id)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / logical_name
    path.write_bytes(content)
    with get_session_factory()() as db:
        db.add(
            Run(
                id=run_id,
                workspace_id=workspace_id,
                module="netrecon",
                task_key=task_key,
                status="succeeded",
                progress=100,
                message_key="run.message.succeeded",
                params_json=json.dumps({"condition": "cond"}),
                finished_at=utcnow(),
            )
        )
        db.add(
            Artifact(
                run_id=run_id,
                workspace_id=workspace_id,
                logical_name=logical_name,
                relative_path=logical_name,
                semantic_type=semantic_type,
                kind="zip",
                mime="application/zip",
                size_bytes=len(content),
                sha256=sha256_file(path),
                is_primary=True,
            )
        )
        db.commit()
    return run_id


def test_derived_charts_single_layer(client):
    workspace_id = _workspace_id(client)
    run_id = _seed_run(
        workspace_id,
        "netrecon.single",
        _zip_bytes(),
        logical_name="single_layer_idopnetwork_export.zip",
        semantic_type="single_layer_idopnetwork_export",
    )

    response = client.post(
        f"/api/runs/{run_id}/derived-charts", json={"kind": "effect"}
    )
    assert response.status_code == 201, response.text
    effect = response.json()
    assert effect["semantic_type"] == "netrecon_derived_effect_html"
    detail = client.get(f"/api/runs/{run_id}").json()
    assert effect["id"] in {item["id"] for item in detail["artifacts"]}
    html = client.get(f"/api/artifacts/{effect['id']}/content").text
    assert "Effect decomposition" in html
    assert "Self-effect + initial value" in html

    response = client.post(
        f"/api/runs/{run_id}/derived-charts",
        json={"kind": "network", "top_edges": 2},
    )
    assert response.status_code == 201, response.text
    network = response.json()
    assert "vis" in client.get(f"/api/artifacts/{network['id']}/content").text.lower()

    response = client.post(
        f"/api/runs/{run_id}/derived-charts", json={"kind": "heatmap"}
    )
    assert response.status_code == 201, response.text
    assert "Adjusted Matrix Heatmap" in client.get(
        f"/api/artifacts/{response.json()['id']}/content"
    ).text

    response = client.post(
        f"/api/runs/{run_id}/derived-charts", json={"kind": "effect", "member": "x"}
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "derived.member_not_supported"


def test_derived_charts_multilayer_member(client, other_client):
    workspace_id = _workspace_id(client)
    run_id = _seed_run(
        workspace_id,
        "netrecon.multi",
        _zip_bytes("intra_cluster/cond/M1"),
        logical_name="multi_layer_idopnetwork_export.zip",
        semantic_type="multi_layer_idopnetwork_export",
    )
    member = "intra_cluster/cond/M1/from_to.csv"
    response = client.post(
        f"/api/runs/{run_id}/derived-charts",
        json={"kind": "effect", "member": member},
    )
    assert response.status_code == 201, response.text
    assert "intra_cluster-cond-M1" in response.json()["logical_name"]

    response = client.post(
        f"/api/runs/{run_id}/derived-charts",
        json={"kind": "network", "member": "missing/from_to.csv"},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "derived.member_not_found"

    other_client.get("/api/session")
    forbidden = other_client.post(
        f"/api/runs/{run_id}/derived-charts", json={"kind": "heatmap"}
    )
    assert forbidden.status_code == 403

    other = _seed_run(
        workspace_id,
        "funclu.k",
        _zip_bytes(),
        logical_name="funclu_k_export.zip",
        semantic_type="funclu_k_export",
    )
    response = client.post(f"/api/runs/{other}/derived-charts", json={"kind": "effect"})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "derived.unsupported_run"


def test_task_gating_and_upload_quota(client):
    from app.jobs.registry import get_task, task_enabled, task_maturity, task_public_spec

    settings = Settings(disabled_tasks="demo.echo", beta_tasks="netanal.glmy")
    assert task_enabled(get_task("curve_fitting.fit"), settings)
    assert not task_enabled(get_task("demo.echo"), settings)
    assert task_maturity(get_task("netanal.glmy"), settings) == "beta"
    assert task_public_spec(get_task("netanal.glmy"), settings)["maturity"] == "beta"

    quota = Settings(max_storage_bytes_per_workspace=10)
    app.dependency_overrides[get_settings] = lambda: quota
    try:
        response = client.post(
            "/api/datasets", files={"file": ("a.csv", b"x" * 100, "text/csv")}
        )
    finally:
        app.dependency_overrides.pop(get_settings, None)
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "storage.quota"
