from __future__ import annotations

import io
import json
import zipfile

from app.job_runner import execute_run

FROM_TO = (
    b"from,to,weight\n"
    b"A,B,1.0\n"
    b"B,C,-0.6\n"
    b"A,C,0.3\n"
    b"C,C,0.2\n"
)


def netrecon_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("from_to.csv", FROM_TO)
        archive.writestr(
            "adjacency_matrix.csv",
            b",A,B,C\nA,0,1.0,0.3\nB,0,0,-0.6\nC,0,0,0.2\n",
        )
    return buffer.getvalue()


def upload_zip(client, content: bytes, name: str = "netrecon.zip") -> dict:
    response = client.post(
        "/api/datasets", files={"file": (name, content, "application/zip")}
    )
    assert response.status_code == 201, response.text
    return response.json()


def create_run(client, params: dict, dataset: dict) -> dict:
    response = client.post(
        "/api/runs",
        json={
            "task_key": "netanal.glmy",
            "params": params,
            "inputs": [
                {"role": "netrecon_zip", "ordinal": 0, "dataset_id": dataset["id"]}
            ],
        },
    )
    assert response.status_code == 202, response.text
    return response.json()


def artifact(detail: dict, logical_name: str) -> dict:
    return next(
        item for item in detail["artifacts"] if item["logical_name"] == logical_name
    )


def test_netanal_glmy_all_edges_end_to_end(client):
    dataset = upload_zip(client, netrecon_zip())
    inspect = client.get(f"/api/datasets/{dataset['id']}/inspect").json()
    assert "from_to.csv" in inspect["members"]

    run = create_run(
        client,
        {
            "member": "from_to.csv",
            "mode": "all_edges_colored",
            "n_stages": 3,
            "max_columns": 2,
            "export_vector": False,
        },
        dataset,
    )
    assert execute_run(run["id"]) == 0
    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["status"] == "succeeded", detail.get("error_text")
    names = {item["logical_name"] for item in detail["artifacts"]}
    assert "glmy_all_edges_colored.zip" in names
    assert "tables/from_to.csv" in names
    assert "netanal_summary.json" in names
    assert any(
        name.startswith("glmy/01_all_edges_colored/") and name.endswith(".png")
        for name in names
    )
    assert any(
        name.startswith("glmy/01_all_edges_colored/") and name.endswith(".csv")
        for name in names
    )

    archive = artifact(detail, "glmy_all_edges_colored.zip")
    download = client.get(f"/api/artifacts/{archive['id']}/download")
    with zipfile.ZipFile(io.BytesIO(download.content)) as zipped:
        members = set(zipped.namelist())
    assert "01_all_edges_colored/filtration_process.png" in members
    assert "01_all_edges_colored/barcode.csv" in members

    summary = json.loads(
        client.get(
            f"/api/artifacts/{artifact(detail, 'netanal_summary.json')['id']}/download"
        ).content
    )
    assert summary["removed_self_loops"] == 1
    assert summary["edges"] == 4
    assert summary["nodes"] == 3


def test_netanal_glmy_separate_signs(client):
    dataset = upload_zip(client, netrecon_zip())
    run = create_run(
        client,
        {
            "member": "from_to.csv",
            "mode": "separate_signs",
            "n_stages": 2,
            "max_columns": 1,
            "export_vector": False,
        },
        dataset,
    )
    assert execute_run(run["id"]) == 0
    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["status"] == "succeeded", detail.get("error_text")
    names = {item["logical_name"] for item in detail["artifacts"]}
    assert "glmy_separate_signs.zip" in names
    assert any(name.startswith("glmy/02_separate_signs/positive/") for name in names)
    assert any(name.startswith("glmy/02_separate_signs/negative/") for name in names)
