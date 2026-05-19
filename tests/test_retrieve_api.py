from fastapi.testclient import TestClient

from graphrag_gnn_qa.api.routes_retrieve import get_vector_retriever
from graphrag_gnn_qa.main import app
from graphrag_gnn_qa.retrieval.vector_retriever import RetrievedChunk


class FakeRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                score=0.91,
                chunk_id="sample_chunk_0000",
                document_id="sample",
                content="GraphRAG connects vector search and graph traversal.",
                source="sample.txt",
                file_name="sample.txt",
                file_type="txt",
            )
        ]


def test_retrieve_chunks() -> None:
    app.dependency_overrides[get_vector_retriever] = lambda: FakeRetriever()
    client = TestClient(app)

    response = client.post("/retrieve", json={"query": "What is GraphRAG?", "top_k": 3})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {
        "query": "What is GraphRAG?",
        "top_k": 3,
        "results": [
            {
                "score": 0.91,
                "chunk_id": "sample_chunk_0000",
                "document_id": "sample",
                "content": "GraphRAG connects vector search and graph traversal.",
                "source": "sample.txt",
                "file_name": "sample.txt",
                "file_type": "txt",
            }
        ],
    }


def test_retrieve_chunks_rejects_invalid_top_k() -> None:
    app.dependency_overrides[get_vector_retriever] = lambda: FakeRetriever()
    client = TestClient(app)

    response = client.post("/retrieve", json={"query": "GraphRAG", "top_k": 0})

    app.dependency_overrides.clear()
    assert response.status_code == 422


def test_retrieve_chunks_rejects_empty_query() -> None:
    app.dependency_overrides[get_vector_retriever] = lambda: FakeRetriever()
    client = TestClient(app)

    response = client.post("/retrieve", json={"query": "", "top_k": 3})

    app.dependency_overrides.clear()
    assert response.status_code == 422
