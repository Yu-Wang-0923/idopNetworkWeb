from __future__ import annotations

import io
import zipfile


def _curve_zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("cond/quasi_dynamic.csv", "quasi time,x,y\n1,1,2\n2,3,4\n")
        archive.writestr("cond/curve_params.csv", "feature,a,b\nx,1,1\n")
        archive.writestr("cond/curve_sample.csv", "quasi time,x,y\n1,1,2\n")
    return buffer.getvalue()


def test_upload_csv_and_inspect(client):
    response = client.post(
        "/api/datasets",
        files={"file": ("scores.csv", b"name,a,b\nx,1,2\ny,3,4\n", "text/csv")},
    )
    assert response.status_code == 201, response.text
    dataset = response.json()
    assert dataset["detected_kind"] == "csv"

    inspected = client.get(f"/api/datasets/{dataset['id']}/inspect").json()
    assert inspected["kind"] == "csv"
    assert inspected["conditions"] == ["scores.csv"]

    listed = client.get("/api/datasets").json()
    assert any(item["id"] == dataset["id"] for item in listed)

    deleted = client.delete(f"/api/datasets/{dataset['id']}")
    assert deleted.status_code == 204
    assert client.get("/api/datasets").json() == []


def test_upload_zip_and_inspect(client):
    response = client.post(
        "/api/datasets",
        files={"file": ("curves.zip", _curve_zip_bytes(), "application/zip")},
    )
    assert response.status_code == 201, response.text
    dataset = response.json()
    assert dataset["detected_kind"] == "curve_fitting_export"
    inspected = client.get(f"/api/datasets/{dataset['id']}/inspect").json()
    assert inspected["conditions"] == ["cond"]


def test_reject_unsupported_extension(client):
    response = client.post(
        "/api/datasets",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_reject_unsafe_zip(client):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../evil.csv", "a\n1\n")
    response = client.post(
        "/api/datasets",
        files={"file": ("evil.zip", buffer.getvalue(), "application/zip")},
    )
    assert response.status_code == 400


def test_other_workspace_cannot_see_dataset(client, other_client):
    response = client.post(
        "/api/datasets",
        files={"file": ("scores.csv", b"name,a\nx,1\n", "text/csv")},
    )
    dataset = response.json()
    assert other_client.get("/api/datasets").json() == []
    assert other_client.get(f"/api/datasets/{dataset['id']}/inspect").status_code == 404
