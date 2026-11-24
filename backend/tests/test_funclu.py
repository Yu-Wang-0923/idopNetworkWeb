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
            (header + rows).encode(), data_mode="quasi_dynamic",
            n_samples=12, trim_percent=0,
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


def create_funclu_run(client, task_key: str, params: dict, dataset: dict) -> dict:
    response = client.post(
        "/api/runs",
        json={
            "task_key": task_key,
            "params": params,
            "inputs": [
                {"role": "curve_zip", "ordinal": 0, "dataset_id": dataset["id"]}
            ],
        },
    )
    assert response.status_code == 202, response.text
    return response.json()


def artifact(detail: dict, logical_name: str) -> dict:
    return next(
        item for item in detail["artifacts"] if item["logical_name"] == logical_name
    )


def test_dataset_inspect_reports_shared_features(client):
    dataset = upload_zip(client, curve_zip_bytes())
    assert dataset["detected_kind"] == "curve_fitting_export"
    inspect = client.get(f"/api/datasets/{dataset['id']}/inspect").json()
    assert inspect["kind"] == "curve_fitting_export"
    assert inspect["conditions"] == ["alpha.csv", "beta.csv"]
    assert inspect["features"] == 4


def test_funclu_k_end_to_end(client):
    dataset = upload_zip(client, curve_zip_bytes())
    run = create_funclu_run(
        client,
        "funclu.k",
        {"n_components": 2, "max_iter": 200, "tol": 1e-4, "random_state": 42},
        dataset,
    )
    assert execute_run(run["id"]) == 0

    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["status"] == "succeeded", detail.get("error_text")
    names = {item["logical_name"] for item in detail["artifacts"]}
    assert {
        "funclu_k_export.zip",
        "charts/cluster_profiles.html",
        "tables/labels.csv",
        "tables/cluster_sizes.csv",
        "tables/mu_params.csv",
        "tables/cov_params.csv",
        "funclu_k_summary.json",
    } <= names

    export = artifact(detail, "funclu_k_export.zip")
    assert export["is_primary"] is True
    download = client.get(f"/api/artifacts/{export['id']}/download")
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        members = set(archive.namelist())
        assert "labels.csv" in members
        assert "cluster_sizes.csv" in members
        assert "cluster_members/alpha.csv/M1_curve_sample.csv" in members
        assert "cluster_members/beta.csv/M2_quasi_dynamic.csv" in members
        assert (
            "cluster_centers/alpha.csv/cluster_center_curve_sample.csv" in members
        )
        labels = pd.read_csv(
            io.BytesIO(archive.read("labels.csv")), dtype={"feature": str}
        )
    assert len(labels) == 4
    assert set(labels["cluster_id"]) == {1, 2}

    chart = artifact(detail, "charts/cluster_profiles.html")
    content = client.get(f"/api/artifacts/{chart['id']}/content")
    assert content.status_code == 200
    assert "sandbox" in content.headers["content-security-policy"]
    assert "M1" in content.text and "M2" in content.text
    assert "alpha.csv" in content.text

    summary = artifact(detail, "funclu_k_summary.json")
    payload = json.loads(
        client.get(f"/api/artifacts/{summary['id']}/download").content
    )
    assert payload["n_components"] == 2
    assert set(payload["cluster_sizes"]) == {"M1", "M2"}


def test_funclu_bic_end_to_end(client):
    dataset = upload_zip(client, curve_zip_bytes())
    run = create_funclu_run(
        client,
        "funclu.bic",
        {
            "k_min": 2,
            "k_max": 3,
            "step": 1,
            "n_init": 1,
            "max_iter": 200,
            "tol": 1e-4,
            "random_state": 42,
            "min_cluster_size": 1,
        },
        dataset,
    )
    assert execute_run(run["id"]) == 0

    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["status"] == "succeeded", detail.get("error_text")
    names = {item["logical_name"] for item in detail["artifacts"]}
    assert {
        "funclu_bic_results.csv",
        "charts/bic_elbow.png",
        "funclu_bic_summary.json",
    } <= names

    csv_artifact = artifact(detail, "funclu_bic_results.csv")
    download = client.get(f"/api/artifacts/{csv_artifact['id']}/download")
    frame = pd.read_csv(io.BytesIO(download.content))
    assert set(frame["K"]) == {2, 3}
    for column in (
        "BIC",
        "AIC",
        "n_parameters",
        "covariance_model",
        "eligible",
        "delta_bic",
        "selected",
    ):
        assert column in frame.columns
    assert int(frame["selected"].sum()) == 1
    selected = frame[frame["selected"]].iloc[0]
    assert selected["delta_bic"] == 0
    assert selected["BIC"] == frame["BIC"].min()

    elbow = artifact(detail, "charts/bic_elbow.png")
    png = client.get(f"/api/artifacts/{elbow['id']}/content")
    assert png.status_code == 200
    assert png.headers["content-type"].startswith("image/png")
    assert png.content.startswith(b"\x89PNG")

    summary = artifact(detail, "funclu_bic_summary.json")
    payload = json.loads(
        client.get(f"/api/artifacts/{summary['id']}/download").content
    )
    assert payload["k_values"] == [2, 3]
    assert payload["recommended_k"] == int(selected["K"])
    assert payload["eligible_count"] == 2
    assert payload["best_row"]["AIC"] is not None


def test_scan_bic_can_select_a_single_module():
    from app.pure.funclu import load_curve_fitting_zip, scan_bic
    from idopnetwork.curve_fitting import get_power_function_params

    curve, quasi = load_curve_fitting_zip(curve_zip_bytes())
    names = list(curve)
    table = scan_bic(
        [curve[name] for name in names],
        [get_power_function_params(quasi[name]) for name in names],
        k_min=1,
        k_max=2,
        step=1,
        n_init=1,
        max_iter=200,
    )
    assert set(table["K"]) == {1, 2}
    assert int(table["selected"].sum()) == 1
    selected = table[table["selected"]].iloc[0]
    assert selected["BIC"] == table["BIC"].min()


def test_funclu_rejects_zip_without_curve_tables(client):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.txt", "not a curve export")
    dataset = upload_zip(client, buffer.getvalue(), "notes.zip")
    run = create_funclu_run(
        client, "funclu.k", {"n_components": 2}, dataset
    )
    assert execute_run(run["id"]) == 1
    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["status"] == "failed"
    assert "curve_sample" in (detail["error_text"] or "")


def test_funclu_bic_cancels_at_k_boundary(client, monkeypatch):
    dataset = upload_zip(client, curve_zip_bytes())
    run = create_funclu_run(
        client,
        "funclu.bic",
        {"k_min": 2, "k_max": 3, "step": 1, "n_init": 1},
        dataset,
    )

    from app.db import get_session_factory
    from app.models import Run

    def fake_scan_bic(*args, progress=None, check=None, **kwargs):
        with get_session_factory()() as db:
            record = db.get(Run, run["id"])
            record.cancel_requested = True
            db.commit()
        progress(2, 1, 2)
        check()
        raise AssertionError("cancellation should stop the scan before K=3")

    monkeypatch.setattr("app.tasks.funclu.scan_bic", fake_scan_bic)
    assert execute_run(run["id"]) == 0
    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["status"] == "cancelled"
    assert not detail["artifacts"]
