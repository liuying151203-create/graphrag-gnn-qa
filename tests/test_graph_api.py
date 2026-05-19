from fastapi.testclient import TestClient

from graphrag_gnn_qa.api.routes_graph import get_graph_retriever
from graphrag_gnn_qa.main import app
from graphrag_gnn_qa.retrieval.graph_retriever import RetrievedGraphRelation


class FakeGraphRetriever:
    def retrieve(self, query: str, top_k: int = 5, max_depth: int = 1) -> list[RetrievedGraphRelation]:
        return [
            RetrievedGraphRelation(
                center_id="Method:graphrag",
                center_name="GraphRAG",
                center_type="Method",
                source_id="Method:graphrag",
                source_name="GraphRAG",
                source_type="Method",
                relation_type="SOLVES_TASK",
                target_id="Task:question answering",
                target_name="question answering",
                target_type="Task",
                chunk_id="sample_chunk_0000",
                document_id="sample",
                source="sample.txt",
                evidence="GraphRAG improves question answering.",
                confidence=0.9,
            )
        ]


def test_retrieve_graph_relations() -> None:
    app.dependency_overrides[get_graph_retriever] = lambda: FakeGraphRetriever()
    client = TestClient(app)

    response = client.post("/graph/retrieve", json={"query": "GraphRAG", "top_k": 3, "max_depth": 2})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {
        "query": "GraphRAG",
        "top_k": 3,
        "max_depth": 2,
        "results": [
            {
                "center_id": "Method:graphrag",
                "center_name": "GraphRAG",
                "center_type": "Method",
                "source_id": "Method:graphrag",
                "source_name": "GraphRAG",
                "source_type": "Method",
                "relation_type": "SOLVES_TASK",
                "target_id": "Task:question answering",
                "target_name": "question answering",
                "target_type": "Task",
                "chunk_id": "sample_chunk_0000",
                "document_id": "sample",
                "source": "sample.txt",
                "evidence": "GraphRAG improves question answering.",
                "confidence": 0.9,
            }
        ],
    }


def test_retrieve_graph_rejects_invalid_top_k() -> None:
    app.dependency_overrides[get_graph_retriever] = lambda: FakeGraphRetriever()
    client = TestClient(app)

    response = client.post("/graph/retrieve", json={"query": "GraphRAG", "top_k": 0, "max_depth": 2})

    app.dependency_overrides.clear()
    assert response.status_code == 422


def test_retrieve_graph_rejects_invalid_max_depth() -> None:
    app.dependency_overrides[get_graph_retriever] = lambda: FakeGraphRetriever()
    client = TestClient(app)

    response = client.post("/graph/retrieve", json={"query": "GraphRAG", "top_k": 3, "max_depth": 0})

    app.dependency_overrides.clear()
    assert response.status_code == 422


def test_retrieve_graph_rejects_empty_query() -> None:
    app.dependency_overrides[get_graph_retriever] = lambda: FakeGraphRetriever()
    client = TestClient(app)

    response = client.post("/graph/retrieve", json={"query": "", "top_k": 3, "max_depth": 2})

    app.dependency_overrides.clear()
    assert response.status_code == 422
