from __future__ import annotations


def test_session_creates_cookie_and_is_stable(client):
    first = client.get("/api/session")
    assert first.status_code == 200
    payload = first.json()
    assert payload["retention_days"] >= 1
    assert payload["max_csv_bytes"] > 0
    assert any(cookie.name == "idop_workspace" for cookie in client.cookies.jar)

    second = client.get("/api/session")
    assert second.status_code == 200
    assert second.json()["created_at"] == payload["created_at"]


def test_session_cookie_is_signed(client):
    client.get("/api/session")
    cookie = next(c for c in client.cookies.jar if c.name == "idop_workspace")
    assert "." in cookie.value
    assert len(cookie.value) > 40
