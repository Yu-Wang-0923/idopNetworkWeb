from __future__ import annotations

import io
import json
import re
import zipfile

import numpy as np
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
        centers = pd.read_csv(
            io.BytesIO(
                archive.read("cluster_centers/alpha.csv/cluster_center_curve_sample.csv")
            ),
            index_col=0,
        )
        observed_centers = pd.read_csv(
            io.BytesIO(
                archive.read("cluster_centers/alpha.csv/cluster_center_quasi_dynamic.csv")
            ),
            index_col=0,
        )
    assert len(labels) == 4
    assert set(labels["cluster_id"]) == {1, 2}
    # Fitted cluster means sit on the 12-point curve_sample grid; the observed
    # member means keep the 8 samples.
    assert list(centers.columns) == ["M1", "M2"]
    assert len(centers) == 12 and np.isfinite(centers.to_numpy()).all()
    assert len(observed_centers) == 8

    chart = artifact(detail, "charts/cluster_profiles.html")
    content = client.get(f"/api/artifacts/{chart['id']}/content")
    assert content.status_code == 200
    assert "sandbox" in content.headers["content-security-policy"]
    assert "M1" in content.text and "M2" in content.text
    assert "alpha.csv" in content.text
    # The figure's mean line is the fitted cluster mean exported as the center.
    match = re.search(r"DATA = (\{.*\});\s*if \(!DATA\) return;", content.text, re.S)
    chart_data = json.loads(match.group(1))
    alpha = next(
        series
        for series in chart_data["panels"][0]["series"]
        if series["label"] == "alpha.csv"
    )
    assert len(alpha["xs"]) == 12 and len(alpha["obs_xs"]) == 8
    np.testing.assert_allclose(alpha["mean"], centers["M1"].to_numpy(), atol=1e-4)

    summary = artifact(detail, "funclu_k_summary.json")
    payload = json.loads(
        client.get(f"/api/artifacts/{summary['id']}/download").content
    )
    assert payload["n_components"] == 2
    assert set(payload["cluster_sizes"]) == {"M1", "M2"}
    assert payload["input_table"] == "quasi_dynamic"
    assert payload["n_points"] == {"alpha.csv": 8, "beta.csv": 8}


def test_funclu_clusters_observations_not_sampled_curves(client, monkeypatch):
    from app.pure.fast_funclu_power import PowerFunClu
    from app.pure.funclu import load_curve_fitting_zip

    seen: list[list[pd.DataFrame]] = []
    original_fit = PowerFunClu.fit

    def spy_fit(self, data, **kwargs):
        seen.append([frame.copy() for frame in data])
        return original_fit(self, data, **kwargs)

    monkeypatch.setattr(PowerFunClu, "fit", spy_fit)
    content = curve_zip_bytes()
    dataset = upload_zip(client, content)
    run = create_funclu_run(
        client, "funclu.k", {"n_components": 2, "max_iter": 200}, dataset
    )
    assert execute_run(run["id"]) == 0

    curve, quasi = load_curve_fitting_zip(content)
    (frames,) = seen
    assert len(frames) == len(quasi)
    for frame, name in zip(frames, quasi):
        pd.testing.assert_frame_equal(frame, quasi[name].loc[:, frame.columns])
        assert len(frame) != len(curve[name])


def test_cluster_panels_use_model_means_and_both_member_tables():
    from app.pure.charts import funclu_html

    grid = [1.0, 2.0, 3.0]
    curves = {"c": pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]}, index=grid)}
    observed = {"c": pd.DataFrame({"a": [1.1, 2.2], "b": [1.9, 3.1]}, index=[1.5, 2.5])}
    means = {"c": np.array([[10.0, 20.0, 30.0]])}
    panels = funclu_html.build_cluster_panels(
        curves, observed, means, np.array([0, 0]), ["a", "b"], 1
    )
    (series,) = panels[0]["series"]
    assert series["xs"] == grid
    assert series["mean"] == [10.0, 20.0, 30.0]
    assert series["curves"] == [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
    assert series["obs_xs"] == [1.5, 2.5]
    assert series["observations"] == [[1.1, 2.2], [1.9, 3.1]]
    html = funclu_html.render(panels, conditions=["c"])
    assert 'id="membermode"' in html


def test_mean_curves_at_matches_the_fitted_means():
    from app.pure.fast_funclu_power import PowerFunClu
    from app.pure.funclu import load_curve_fitting_zip

    _curve, quasi = load_curve_fitting_zip(curve_zip_bytes())
    frames = list(quasi.values())
    model = PowerFunClu(n_components=2, max_iter=200).fit(frames)
    for index, frame in enumerate(frames):
        expected = model.model_.mean_curves_[str(index)].to_numpy().T
        np.testing.assert_allclose(model.mean_curves_at(index, frame.index), expected)


def test_funclu_bic_end_to_end(client):
    dataset = upload_zip(client, curve_zip_bytes())
    run = create_funclu_run(
        client,
        "funclu.bic",
        {
            "k_min": 2,
            "k_max": 3,
            "step": 1,
            "covariance_models": ["S1", "S4"],
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
        "charts/bic_comparison.png",
        "funclu_bic_summary.json",
    } <= names

    csv_artifact = artifact(detail, "funclu_bic_results.csv")
    download = client.get(f"/api/artifacts/{csv_artifact['id']}/download")
    assert b"\r\r\n" not in download.content
    frame = pd.read_csv(io.BytesIO(download.content))
    assert set(frame["K"]) == {2, 3}
    assert set(frame["covariance_model"]) == {"S1", "S4"}
    assert len(frame) == 4
    for column in (
        "BIC",
        "AIC",
        "n_parameters",
        "n_covariance_parameters",
        "eligible",
        "delta_bic",
        "selected",
    ):
        assert column in frame.columns
    assert int(frame["selected"].sum()) == 1
    selected = frame[frame["selected"]].iloc[0]
    assert selected["delta_bic"] == 0
    eligible = frame[frame["eligible"].astype(bool)]
    assert selected["BIC"] == eligible["BIC"].min()

    comparison = artifact(detail, "charts/bic_comparison.png")
    png = client.get(f"/api/artifacts/{comparison['id']}/content")
    assert png.status_code == 200
    assert png.headers["content-type"].startswith("image/png")
    assert png.content.startswith(b"\x89PNG")

    summary = artifact(detail, "funclu_bic_summary.json")
    payload = json.loads(
        client.get(f"/api/artifacts/{summary['id']}/download").content
    )
    assert payload["k_values"] == [2, 3]
    assert payload["covariance_models"] == ["S1", "S4"]
    assert payload["recommended_k"] == int(selected["K"])
    assert payload["selected_model"] == str(selected["covariance_model"])
    assert payload["eligible_count"] >= 1
    assert payload["best_row"]["AIC"] is not None
    assert payload["best_row"]["covariance_model"] == payload["selected_model"]

    # 4 shared features scanned at K = 2..3: either end can still be widened.
    side = "upper" if payload["recommended_k"] == 3 else "lower"
    assert payload["boundary_side"] == side
    assert payload["selected_at_boundary"] is True
    logs = client.get(f"/api/runs/{run['id']}/logs").json()
    assert any(
        entry["level"] == "warning"
        and entry["message_key"] == f"funclu.message.bic_boundary_{side}"
        and entry["message_params"]["k"] == payload["recommended_k"]
        for entry in logs
    )


def test_bic_boundary_side_only_flags_extendable_ends():
    from app.tasks.funclu import bic_boundary_side

    assert bic_boundary_side([2, 3, 4], 4, n_features=65) == "upper"
    assert bic_boundary_side([2, 3, 4], 2, n_features=65) == "lower"
    assert bic_boundary_side([2, 3, 4], 3, n_features=65) is None
    assert bic_boundary_side([1, 2, 3], 1, n_features=65) is None
    assert bic_boundary_side([2, 3, 4], 4, n_features=4) is None
    assert bic_boundary_side([3], 3, n_features=65) is None


def test_funclu_k_supports_shared_covariance(client):
    dataset = upload_zip(client, curve_zip_bytes())
    run = create_funclu_run(
        client,
        "funclu.k",
        {
            "n_components": 2,
            "max_iter": 200,
            "tol": 1e-4,
            "random_state": 42,
            "covariance_model": "S1",
        },
        dataset,
    )
    assert execute_run(run["id"]) == 0
    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["status"] == "succeeded", detail.get("error_text")
    summary = artifact(detail, "funclu_k_summary.json")
    payload = json.loads(
        client.get(f"/api/artifacts/{summary['id']}/download").content
    )
    assert payload["covariance_model"] == "S1"
    assert payload["covariance_spec"]["n_covariance_parameters"] == 2


def test_scan_bic_can_select_a_single_module():
    from app.pure.funclu import load_curve_fitting_zip, scan_bic

    _curve, quasi = load_curve_fitting_zip(curve_zip_bytes())
    names = list(quasi)
    table = scan_bic(
        [quasi[name] for name in names],
        k_min=1,
        k_max=2,
        step=1,
        covariance_models=("S4",),
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
