from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from app.job_runner import execute_run
from app.pure.charts import curve_fitting_html
from app.pure.curve_fitting import build_fitting_export, fit_uploaded_csv

FIXTURES = Path(__file__).parent / "fixtures"


def upload_csv(client, name: str, content: bytes) -> dict:
    response = client.post(
        "/api/datasets",
        files={"file": (name, content, "text/csv")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def create_run(client, params: dict, datasets: list[dict]) -> dict:
    response = client.post(
        "/api/runs",
        json={
            "task_key": "curve_fitting.fit",
            "params": params,
            "inputs": [
                {"role": "csv", "ordinal": index, "dataset_id": dataset["id"]}
                for index, dataset in enumerate(datasets)
            ],
        },
    )
    assert response.status_code == 202, response.text
    return response.json()


def artifact(detail: dict, logical_name: str) -> dict:
    return next(
        item for item in detail["artifacts"] if item["logical_name"] == logical_name
    )


def test_quasi_fit_and_export_contract():
    content = (
        b"sample,a,b,c\n"
        b"s1,1,2,3\n"
        b"s2,2,4,6\n"
        b"s3,3,6,9\n"
        b"s4,4,8,12\n"
        b"s5,5,10,15\n"
    )
    tables = fit_uploaded_csv(
        content, data_mode="quasi_dynamic", n_samples=10, trim_percent=0
    )
    assert set(tables) == {"quasi_dynamic", "curve_params", "curve_sample"}
    assert list(tables["curve_params"].columns) == ["c", "beta", "a", "b", "n_positive"]
    assert len(tables["curve_sample"]) == 10
    assert len(tables["quasi_dynamic"]) == 5

    blob = build_fitting_export({"sample.csv": tables})
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        assert set(archive.namelist()) == {
            "sample.csv/quasi_dynamic.csv",
            "sample.csv/curve_params.csv",
            "sample.csv/curve_sample.csv",
        }
        frame = pd.read_csv(
            io.BytesIO(archive.read("sample.csv/curve_sample.csv")), index_col=0
        )
        assert frame.shape == (10, 3)


def test_renderer_escapes_feature_names():
    feature = "<script>alert(1)</script>"
    frame = pd.DataFrame({feature: [1.0, 2.0, 3.0]}, index=[1.0, 2.0, 3.0])
    html = curve_fitting_html.render(
        curve_fitting_html.build_panels(frame, frame, [feature])
    )
    assert "<script>alert(1)</script>" not in html
    assert "\\u003cscript" in html


def test_quasi_end_to_end(client):
    alpha = b"sample,a,b\ns1,1,2\ns2,2,4\ns3,4,8\ns4,8,16\ns5,16,32\n"
    beta = b"sample,a,b\ns1,2,1\ns2,4,2\ns3,8,4\ns4,16,8\ns5,32,16\n"
    first = upload_csv(client, "alpha.csv", alpha)
    second = upload_csv(client, "beta.csv", beta)

    run = create_run(
        client,
        {"data_mode": "quasi_dynamic", "n_samples": 12, "trim_percent": 0},
        [first, second],
    )
    assert execute_run(run["id"]) == 0

    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["status"] == "succeeded", detail.get("error_text")
    names = {item["logical_name"] for item in detail["artifacts"]}
    assert "curve_fitting_export.zip" in names
    assert "charts/alpha/curve_fitting.html" in names
    assert "charts/beta/curve_fitting.html" in names
    assert "charts/compare.html" in names
    assert "tables/alpha_params.csv" in names
    assert "tables/beta_samples.csv" in names
    assert "curve_fitting_summary.json" in names

    archive_artifact = artifact(detail, "curve_fitting_export.zip")
    download = client.get(f"/api/artifacts/{archive_artifact['id']}/download")
    assert download.status_code == 200
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        names = set(archive.namelist())
        assert "alpha.csv/quasi_dynamic.csv" in names
        assert "beta.csv/curve_sample.csv" in names

    chart = artifact(detail, "charts/alpha/curve_fitting.html")
    content = client.get(f"/api/artifacts/{chart['id']}/content")
    assert content.status_code == 200
    assert "text/html" in content.headers["content-type"]
    assert "sandbox" in content.headers["content-security-policy"]
    assert "panels" in content.text


def test_duplicate_condition_names_fail(client):
    content = b"sample,a\ns1,1\ns2,2\ns3,3\n"
    first = upload_csv(client, "same.csv", content)
    second = upload_csv(client, "same.csv", content)
    run = create_run(client, {"data_mode": "quasi_dynamic"}, [first, second])
    assert execute_run(run["id"]) == 1
    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["status"] == "failed"
    assert "Duplicate" in (detail["error_text"] or "")


@pytest.mark.slow
def test_dynamic_ecg_end_to_end(client):
    content = (FIXTURES / "ecg_100hz.csv").read_bytes()
    dataset = upload_csv(client, "ecg_100hz.csv", content)
    run = create_run(
        client,
        {
            "data_mode": "dynamic",
            "swt_wavelet": "rbio3.9",
            "swt_level": 2,
            "swt_alpha": 0.5,
            "ecg_sampling_rate": 100,
        },
        [dataset],
    )
    assert execute_run(run["id"]) == 0

    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["status"] == "succeeded", detail.get("error_text")
    assert artifact(detail, "curve_fitting_export.zip")["kind"] == "zip"
    chart = artifact(detail, "charts/ecg_100hz/ecg_fitting.html")
    assert chart["kind"] == "html"
    content = client.get(f"/api/artifacts/{chart['id']}/content")
    assert content.status_code == 200
    assert "Lead I · Full record" in content.text
    assert "Lead I · First 250 samples" in content.text
    assert "Time (s)" in content.text
    assert "Signal (stored units)" in content.text
    assert "#d73027" in content.text
    assert not any(
        item["semantic_type"] == "curve_fitting_ecg"
        for item in detail["artifacts"]
    )
