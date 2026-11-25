from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.job_runner import execute_run
from app.pure.charts import curve_fitting_html
from app.pure.curve_fitting import build_fitting_export, fit_uploaded_csv
from app.pure.funclu import observed_condition_frames

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


ZERO_CSV = b"sample,a,b\ns1,0,1\ns2,2,2\ns3,3,3\ns4,4,4\ns5,5,5\n"


def test_mask_zeros_excludes_raw_zeros_from_the_fit():
    plain = fit_uploaded_csv(ZERO_CSV, n_samples=10, trim_percent=0)
    masked = fit_uploaded_csv(ZERO_CSV, n_samples=10, trim_percent=0, mask_zeros=True)
    assert "masked" not in plain
    # Z_min_add1 then Log10_1p turns the raw 0 into log10(2) > 0, so it is fitted
    # unless masked.
    assert plain["curve_params"].loc["a", "n_positive"] == 5
    assert masked["curve_params"].loc["a", "n_positive"] == 4
    assert masked["curve_params"].loc["b", "n_positive"] == 5
    flags = masked["masked"]
    assert flags.index.equals(masked["quasi_dynamic"].index)
    (row,) = np.flatnonzero(flags["a"].to_numpy())
    assert masked["quasi_dynamic"]["a"].iloc[row] == pytest.approx(np.log10(2))
    assert not flags["b"].any()
    pd.testing.assert_frame_equal(masked["quasi_dynamic"], plain["quasi_dynamic"])


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

    table = artifact(detail, "tables/alpha_params.csv")
    raw = client.get(f"/api/artifacts/{table['id']}/download").content
    assert b"\r\r\n" not in raw
    assert len(pd.read_csv(io.BytesIO(raw))) == 2


def test_mask_zeros_end_to_end(client):
    datasets = [
        upload_csv(client, "zeros.csv", ZERO_CSV),
        upload_csv(client, "zeros2.csv", ZERO_CSV),
    ]
    run = create_run(
        client,
        {
            "data_mode": "quasi_dynamic",
            "n_samples": 10,
            "trim_percent": 0,
            "mask_zeros": True,
        },
        datasets,
    )
    assert execute_run(run["id"]) == 0
    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["status"] == "succeeded", detail.get("error_text")

    summary = artifact(detail, "curve_fitting_summary.json")
    payload = json.loads(client.get(f"/api/artifacts/{summary['id']}/download").content)
    assert payload["params"]["mask_zeros"] is True
    assert [item["masked_points"] for item in payload["files"]] == [1, 1]

    for name in ("charts/zeros/curve_fitting.html", "charts/compare.html"):
        chart = artifact(detail, name)
        html = client.get(f"/api/artifacts/{chart['id']}/content").text
        assert '"masked": [0]' in html, name


SPARSE_CSV = (
    b"sample,dense,sparse\n"
    b"s1,1,0\n"
    b"s2,2,0\n"
    b"s3,3,1\n"
    b"s4,4,0\n"
    b"s5,5,2\n"
    b"s6,6,0\n"
    b"s7,7,3\n"
    b"s8,8,0\n"
)


def test_min_nonzero_leaves_sparse_features_out_of_the_fit():
    plain = fit_uploaded_csv(SPARSE_CSV, n_samples=10, trim_percent=0)
    tables = fit_uploaded_csv(SPARSE_CSV, n_samples=10, trim_percent=0, min_nonzero=6)
    assert "below_min_nonzero" not in plain
    assert tables["below_min_nonzero"].to_dict() == {"dense": False, "sparse": True}
    # No fitted parameters and no curve, so the feature drops out of the figures and
    # FunClu the same way other features that cannot be fitted do.
    assert tables["curve_params"].loc["sparse", "n_positive"] == 0
    assert tables["curve_params"].loc["sparse", ["a", "b"]].isna().all()
    assert tables["curve_sample"]["sparse"].isna().all()
    assert np.isfinite(tables["curve_sample"]["dense"]).all()
    # The observations are unchanged, so the feature stays in quasi_dynamic.
    pd.testing.assert_frame_equal(tables["quasi_dynamic"], plain["quasi_dynamic"])
    assert np.isfinite(plain["curve_sample"]["sparse"]).all()


def test_excluded_features_are_left_out_of_funclu_input():
    names = ["first", "second"]
    fitted = {
        name: fit_uploaded_csv(SPARSE_CSV, n_samples=10, trim_percent=0, min_nonzero=6)
        for name in names
    }
    _, valid = observed_condition_frames(
        {name: fitted[name]["quasi_dynamic"] for name in names},
        {name: fitted[name]["curve_sample"] for name in names},
        names,
    )
    assert valid == ["dense"]


def test_min_nonzero_end_to_end(client):
    datasets = [
        upload_csv(client, "sparse_a.csv", SPARSE_CSV),
        upload_csv(client, "sparse_b.csv", SPARSE_CSV),
    ]
    run = create_run(
        client,
        {
            "data_mode": "quasi_dynamic",
            "n_samples": 10,
            "trim_percent": 0,
            "min_nonzero": 6,
        },
        datasets,
    )
    assert execute_run(run["id"]) == 0
    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["status"] == "succeeded", detail.get("error_text")

    summary = artifact(detail, "curve_fitting_summary.json")
    payload = json.loads(client.get(f"/api/artifacts/{summary['id']}/download").content)
    assert payload["params"]["min_nonzero"] == 6
    assert [item["below_min_nonzero"] for item in payload["files"]] == [["sparse"], ["sparse"]]
    assert [item["features_fitted"] for item in payload["files"]] == [1, 1]

    for name in ("charts/sparse_a/curve_fitting.html", "charts/compare.html"):
        chart = artifact(detail, name)
        html = client.get(f"/api/artifacts/{chart['id']}/content").text
        assert '"name": "dense"' in html, name
        assert '"name": "sparse"' not in html, name


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
