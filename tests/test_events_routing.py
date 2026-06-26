import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(main, "DB_PATH", tmp_path / "routing.db")
    from fastapi.testclient import TestClient
    return TestClient(main.app)


def test_default_view_is_week(client):
    r = client.get("/events")
    assert r.status_code == 200
    assert "view-switch" in r.text  # 切换器存在（Task 10 起）


@pytest.mark.parametrize("view", ["week", "month", "list", "free", "feed"])
def test_all_views_render_200(client, view):
    assert client.get(f"/events?view={view}").status_code == 200


def test_invalid_view_falls_back(client):
    r = client.get("/events?view=bogus")
    assert r.status_code == 200


def test_api_feed_shape(client):
    r = client.get("/api/feed")
    assert r.status_code == 200
    assert "feed" in r.json()
