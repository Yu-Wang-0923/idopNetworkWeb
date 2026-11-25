from __future__ import annotations

import io
import json
import zipfile

import pandas as pd

from app.job_runner import execute_run
from app.pure.curve_fitting import build_fitting_export, fit_uploaded_csv


def curve_zip_bytes() -> bytes:
    def tables(scale: float):
        header = "sample,a,b,c,d\n"
        rows = "".join(
            f"s{i},{i * scale:.4f},{i * 2 * scale:.4f},"
            f"{i * 3 * scale:.4f},{i * 0.5 * scale:.4f}\n"
            for i in range(1, 9)
        )
        return fit_uploaded_csv(
            (header + rows).encode(),
            data_mode="quasi_dynamic",
            n_samples=12,
            trim_percent=0,
        )

    return build_fitting_export(
        {"alpha.csv": tables(1.0), "beta.csv": tables(1.5)}
    )


def upload_zip(client, content: bytes, name: str = "curves.zip") -> dict:
    response = client.post(
        "/api/datasets",
        files={"file": (name, content, "application/zip")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def create_run(client, params: dict, dataset: dict) -> dict:
    response = client.post(
        "/api/runs",
        json={
            "task_key": "netrecon.single",
            "params": params,
            "inputs": [
                {"role": "curve_zip", "ordinal": 0, "dataset_id": dataset["id"]}
            ],
        },
    )
    assert response.status_code == 202, response.text
    return response.json()


def create_funclu_run(client, dataset: dict) -> dict:
    response = client.post(
        "/api/runs",
        json={
            "task_key": "funclu.k",
            "params": {
                "n_components": 2,
                "max_iter": 200,
                "tol": 1e-4,
                "random_state": 42,
            },
            "inputs": [
                {"role": "curve_zip", "ordinal": 0, "dataset_id": dataset["id"]}
            ],
        },
    )
    assert response.status_code == 202, response.text
    return response.json()


def create_multi_run(client, params: dict, artifact_id: str) -> dict:
    response = client.post(
        "/api/runs",
        json={
            "task_key": "netrecon.multi",
            "params": params,
            "inputs": [
                {"role": "funclu_zip", "ordinal": 0, "artifact_id": artifact_id}
            ],
        },
    )
    assert response.status_code == 202, response.text
    return response.json()


def artifact(detail: dict, logical_name: str) -> dict:
    return next(
        item for item in detail["artifacts"] if item["logical_name"] == logical_name
    )


def assert_network_artifacts(client, detail: dict) -> None:
    names = {item["logical_name"] for item in detail["artifacts"]}
    assert {
        "single_layer_idopnetwork_export.zip",
        "charts/network.html",
        "charts/effect_decomposition.html",
        "charts/adjacency_heatmap.html",
        "13A_ring_layout.csv",
        "13B_display_edges.csv",
        "effect_decomposition.zip",
        "tables/adjacency_matrix.csv",
        "tables/from_to.csv",
        "tables/predictions.csv",
        "tables/observations.csv",
        "netrecon_summary.json",
    } <= names

    archive = artifact(detail, "single_layer_idopnetwork_export.zip")
    download = client.get(f"/api/artifacts/{archive['id']}/download")
    with zipfile.ZipFile(io.BytesIO(download.content)) as zipped:
        members = set(zipped.namelist())
        assert "adjacency_matrix.csv" in members
        assert "from_to.csv" in members
        assert "params.csv" in members
        assert "intercepts.csv" in members
        assert any(name.startswith("effect/") for name in members)

    network = artifact(detail, "charts/network.html")
    html = client.get(f"/api/artifacts/{network['id']}/content").text
    assert "vis-network" in html
    assert "unpkg.com" not in html

    effects = artifact(detail, "charts/effect_decomposition.html")
    assert "Effect decomposition" in client.get(
        f"/api/artifacts/{effects['id']}/content"
    ).text

    heatmap = artifact(detail, "charts/adjacency_heatmap.html")
    assert "Adjusted Matrix Heatmap" in client.get(
        f"/api/artifacts/{heatmap['id']}/content"
    ).text


def test_netrecon_single_asgl_end_to_end(client):
    dataset = upload_zip(client, curve_zip_bytes())
    run = create_run(
        client,
        {
            "algorithm": "asgl",
            "condition": "alpha.csv",
            "max_order": 1,
            "nonneg_self": True,
            "adjacency_aggregation": "mean",
        },
        dataset,
    )
    assert execute_run(run["id"]) == 0
    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["status"] == "succeeded", detail.get("error_text")
    assert_network_artifacts(client, detail)
    summary = json.loads(
        client.get(
            f"/api/artifacts/{artifact(detail, 'netrecon_summary.json')['id']}/download"
        ).content
    )
    assert summary["condition"] == "alpha.csv"
    assert summary["algorithm"] == "asgl"
    assert summary["n_features"] == 4


def test_netrecon_single_static_end_to_end(client):
    dataset = upload_zip(client, curve_zip_bytes())
    run = create_run(
        client,
        {
            "algorithm": "static_idop",
            "condition": "beta.csv",
            "max_order": 2,
            "lasso_alpha": 0.1,
            "lasso_windows": 1,
            "lasso_threshold": 0.5,
            "adjacency_aggregation": "integral",
        },
        dataset,
    )
    assert execute_run(run["id"]) == 0
    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["status"] == "succeeded", detail.get("error_text")
    assert_network_artifacts(client, detail)


def test_netrecon_single_dynamic_end_to_end(client):
    dataset = upload_zip(client, curve_zip_bytes())
    run = create_run(
        client,
        {
            "algorithm": "dynamic_idop",
            "condition": "alpha.csv",
            "max_order": 2,
            "adjacency_aggregation": "energy_share",
            "n_fourier": 0,
            "dynamic_fs": 100,
            "lasso_alpha": 0.1,
            "lasso_threshold": 0.9,
        },
        dataset,
    )
    assert execute_run(run["id"]) == 0
    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["status"] == "succeeded", detail.get("error_text")
    assert_network_artifacts(client, detail)
    summary = json.loads(
        client.get(
            f"/api/artifacts/{artifact(detail, 'netrecon_summary.json')['id']}/download"
        ).content
    )
    assert summary["algorithm"] == "dynamic_idop"
    assert summary["solver_version"]


def test_netrecon_multi_end_to_end(client):
    dataset = upload_zip(client, curve_zip_bytes())
    funclu_run = create_funclu_run(client, dataset)
    assert execute_run(funclu_run["id"]) == 0
    funclu_detail = client.get(f"/api/runs/{funclu_run['id']}").json()
    assert funclu_detail["status"] == "succeeded", funclu_detail.get("error_text")
    export = artifact(funclu_detail, "funclu_k_export.zip")

    run = create_multi_run(
        client,
        {
            "algorithm": "asgl",
            "max_order": 1,
            "nonneg_self": True,
            "adjacency_aggregation": "mean",
            "scope": "all",
            "clusters": [],
            "member_cap": 0,
            "top_edges": 0,
        },
        export["id"],
    )
    assert execute_run(run["id"]) == 0
    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["status"] == "succeeded", detail.get("error_text")
    names = {item["logical_name"] for item in detail["artifacts"]}
    assert "multi_layer_idopnetwork_export.zip" in names
    assert "charts/multilayer_overview.html" in names
    assert "tables/network_summary.csv" in names
    assert "multilayer_summary.json" in names
    assert any(name.startswith("charts/networks/inter/") for name in names)
    assert any(name.startswith("charts/networks/intra/") for name in names)

    archive = artifact(detail, "multi_layer_idopnetwork_export.zip")
    download = client.get(f"/api/artifacts/{archive['id']}/download")
    with zipfile.ZipFile(io.BytesIO(download.content)) as zipped:
        members = set(zipped.namelist())
        assert "metadata/labels.csv" in members
        assert any(
            name.startswith("inter_cluster/") and name.endswith("adjacency_matrix.csv")
            for name in members
        )
        assert any(
            name.startswith("intra_cluster/") and name.endswith("adjacency_matrix.csv")
            for name in members
        )
        assert any(name.endswith("/intercepts.csv") for name in members)

    overview = artifact(detail, "charts/multilayer_overview.html")
    html = client.get(f"/api/artifacts/{overview['id']}/content").text
    assert "Multilayer IdopNetwork overview" in html
    assert "Inter-module" in html

    summary = json.loads(
        client.get(
            f"/api/artifacts/{artifact(detail, 'multilayer_summary.json')['id']}/download"
        ).content
    )
    assert summary["scope"] == "all"
    assert summary["n_inter_networks"] >= 1
    assert summary["n_intra_networks"] >= 1

    table = client.get(
        f"/api/artifacts/{artifact(detail, 'tables/network_summary.csv')['id']}/download"
    ).content
    frame = pd.read_csv(io.BytesIO(table))
    assert {"layer", "condition", "cluster", "n_nodes", "n_edges"} <= set(
        frame.columns
    )
