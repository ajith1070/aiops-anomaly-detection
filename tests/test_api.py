import pytest

from app.main import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "healthy"


def test_normal_prediction(client):
    response = client.post(
        "/predict",
        json={"cpu_percent": 30, "memory_percent": 42},
    )
    assert response.status_code == 200
    assert response.json["status"] == "NORMAL"


def test_anomaly_prediction(client):
    response = client.post(
        "/predict",
        json={"cpu_percent": 98, "memory_percent": 95},
    )
    assert response.status_code == 200
    assert response.json["status"] == "ANOMALY"


def test_missing_metrics(client):
    response = client.post("/predict", json={"cpu_percent": 30})
    assert response.status_code == 400


def test_invalid_json_body(client):
    response = client.post("/predict", data="not-json")
    assert response.status_code == 400


def test_prometheus_metrics(client):
    client.post(
        "/predict",
        json={"cpu_percent": 30, "memory_percent": 42},
    )
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"aiops_predictions_total" in response.data
