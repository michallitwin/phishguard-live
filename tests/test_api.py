import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

def test_score_valid_domain():
    response = client.post("/api/score", json={"domain": "paypal-verify-login.tk"})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in {"phishing", "legit"}
    assert 0 <= data['phishing_probability'] <= 1

@pytest.mark.parametrize("domain", ["abc", "not_a_domain", "-bad-.com"])
def test_score_invalid_domain_returns_422(domain):
    response = client.post("/api/score", json={"domain": domain})
    assert response.status_code == 422

def test_metrics_endpoint():
    response = client.get("/api/model/metrics")
    assert response.status_code == 200
    assert 0 <= response.json()["roc_auc"] <= 1
