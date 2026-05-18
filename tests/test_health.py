from fastapi.testclient import TestClient

from graphrag_gnn_qa.main import app


def test_health_check() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app_name": "graphrag-gnn-qa",
        "environment": "development",
    }
