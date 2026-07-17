from fastapi.testclient import TestClient

from graphrag_gnn_qa.api.dependencies import get_runtime_resources
from graphrag_gnn_qa.main import app
from graphrag_gnn_qa.runtime import ComponentReadiness, RuntimeReadiness


class FakeRuntimeResources:
    def __init__(self, readiness: RuntimeReadiness) -> None:
        self._readiness = readiness

    def readiness(self) -> RuntimeReadiness:
        return self._readiness


def test_health_check() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app_name": "graphrag-gnn-qa",
        "environment": "development",
    }


def test_readiness_check_returns_component_statuses() -> None:
    readiness = _build_readiness()
    app.dependency_overrides[get_runtime_resources] = lambda: FakeRuntimeResources(readiness)
    client = TestClient(app)

    response = client.get("/ready")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "components": {
            "api": {"status": "ready", "detail": "FastAPI runtime initialized"},
            "embedding": {"status": "ready", "detail": "BAAI/bge-m3"},
            "milvus": {"status": "ready", "detail": "collection=rag_chunks"},
            "neo4j": {"status": "ready", "detail": "database=neo4j"},
            "reranker": {"status": "ready", "detail": "keyword"},
            "llm": {"status": "ready", "detail": "deepseek-chat"},
        },
    }


def test_readiness_check_returns_503_when_a_component_is_not_ready() -> None:
    readiness = _build_readiness()
    readiness = RuntimeReadiness(
        status="degraded",
        components={
            **readiness.components,
            "llm": ComponentReadiness(
                status="not_configured",
                detail="LLM_API_KEY is not configured",
            ),
        },
    )
    app.dependency_overrides[get_runtime_resources] = lambda: FakeRuntimeResources(readiness)
    client = TestClient(app)

    response = client.get("/ready")

    app.dependency_overrides.clear()
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["components"]["llm"] == {
        "status": "not_configured",
        "detail": "LLM_API_KEY is not configured",
    }


def _build_readiness() -> RuntimeReadiness:
    return RuntimeReadiness(
        status="ready",
        components={
            "api": ComponentReadiness(status="ready", detail="FastAPI runtime initialized"),
            "embedding": ComponentReadiness(status="ready", detail="BAAI/bge-m3"),
            "milvus": ComponentReadiness(status="ready", detail="collection=rag_chunks"),
            "neo4j": ComponentReadiness(status="ready", detail="database=neo4j"),
            "reranker": ComponentReadiness(status="ready", detail="keyword"),
            "llm": ComponentReadiness(status="ready", detail="deepseek-chat"),
        },
    )
