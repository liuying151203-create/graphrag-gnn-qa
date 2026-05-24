from fastapi.testclient import TestClient

from graphrag_gnn_qa.api.routes_debug import get_graph_retriever, get_vector_retriever
from graphrag_gnn_qa.main import app
from graphrag_gnn_qa.retrieval.graph_retriever import RetrievedGraphRelation
from graphrag_gnn_qa.retrieval.vector_retriever import RetrievedChunk


class FakeVectorRetriever:
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


class FakeGraphRetriever:
    def retrieve(self, query: str, top_k: int = 5, max_depth: int = 1) -> list[RetrievedGraphRelation]:
        if query != "GraphRAG":
            return []
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


def test_debug_retrieval_returns_vector_and_graph_results() -> None:
    app.dependency_overrides[get_vector_retriever] = lambda: FakeVectorRetriever()
    app.dependency_overrides[get_graph_retriever] = lambda: FakeGraphRetriever()
    client = TestClient(app)

    response = client.post(
        "/retrieval/debug",
        json={"query": "What is GraphRAG?", "vector_top_k": 3, "graph_top_k": 5, "graph_max_depth": 2},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {
        "query": "What is GraphRAG?",
        "vector_top_k": 3,
        "graph_top_k": 5,
        "graph_max_depth": 2,
        "graph_query_terms": ["What is GraphRAG?", "GraphRAG"],
        "vector_results": [
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
        "graph_results": [
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
        "hybrid_results": [
            {
                "evidence_id": "V1+G1",
                "evidence_type": "hybrid",
                "rank": 1,
                "score": 0.91,
                "fusion_score": 0.937,
                "document_id": "sample",
                "chunk_id": "sample_chunk_0000",
                "source": "sample.txt",
                "content": "GraphRAG connects vector search and graph traversal.\nGraphRAG improves question answering.",
                "metadata": {
                    "file_name": "sample.txt",
                    "file_type": "txt",
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
                    "evidence_ids": "V1,G1",
                    "evidence_types": "vector_chunk,graph_relation",
                    "source_evidence_count": "2",
                },
            },
        ],
    }


def test_debug_retrieval_rejects_invalid_vector_top_k() -> None:
    app.dependency_overrides[get_vector_retriever] = lambda: FakeVectorRetriever()
    app.dependency_overrides[get_graph_retriever] = lambda: FakeGraphRetriever()
    client = TestClient(app)

    response = client.post("/retrieval/debug", json={"query": "GraphRAG", "vector_top_k": 0})

    app.dependency_overrides.clear()
    assert response.status_code == 422


def test_debug_retrieval_rejects_invalid_graph_top_k() -> None:
    app.dependency_overrides[get_vector_retriever] = lambda: FakeVectorRetriever()
    app.dependency_overrides[get_graph_retriever] = lambda: FakeGraphRetriever()
    client = TestClient(app)

    response = client.post("/retrieval/debug", json={"query": "GraphRAG", "graph_top_k": 0})

    app.dependency_overrides.clear()
    assert response.status_code == 422


def test_debug_retrieval_rejects_invalid_graph_max_depth() -> None:
    app.dependency_overrides[get_vector_retriever] = lambda: FakeVectorRetriever()
    app.dependency_overrides[get_graph_retriever] = lambda: FakeGraphRetriever()
    client = TestClient(app)

    response = client.post("/retrieval/debug", json={"query": "GraphRAG", "graph_max_depth": 0})

    app.dependency_overrides.clear()
    assert response.status_code == 422


def test_debug_retrieval_rejects_empty_query() -> None:
    app.dependency_overrides[get_vector_retriever] = lambda: FakeVectorRetriever()
    app.dependency_overrides[get_graph_retriever] = lambda: FakeGraphRetriever()
    client = TestClient(app)

    response = client.post("/retrieval/debug", json={"query": ""})

    app.dependency_overrides.clear()
    assert response.status_code == 422
