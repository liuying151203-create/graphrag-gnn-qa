from fastapi.testclient import TestClient

from graphrag_gnn_qa.api.routes_qa import build_evidence_reranker, get_qa_service
from graphrag_gnn_qa.config import Settings
from graphrag_gnn_qa.main import app
from graphrag_gnn_qa.rag.qa_service import CitationEvidence, GraphEvidence, QAResult, SourceEvidence
from graphrag_gnn_qa.rerank import FallbackEvidenceReranker, KeywordOverlapEvidenceReranker


class FakeQAService:
    def answer(self, question: str, top_k: int = 5) -> QAResult:
        return QAResult(
            question=question,
            answer="GraphRAG combines retrieval and generation.",
            sources=[
                SourceEvidence(
                    chunk_id="sample_chunk_0000",
                    document_id="sample",
                    source="sample.txt",
                    file_name="sample.txt",
                    score=0.91,
                    content="GraphRAG connects vector search and graph traversal.",
                )
            ],
            graph_sources=[
                GraphEvidence(
                    center_name="GraphRAG",
                    center_type="Method",
                    source_name="GraphRAG",
                    source_type="Method",
                    relation_type="SOLVES_TASK",
                    target_name="question answering",
                    target_type="Task",
                    chunk_id="sample_chunk_0000",
                    document_id="sample",
                    source="sample.txt",
                    evidence="GraphRAG improves question answering.",
                    confidence=0.9,
                )
            ],
            citations=[
                CitationEvidence(
                    evidence_id="V1+G1",
                    evidence_type="hybrid",
                    document_id="sample",
                    chunk_id="sample_chunk_0000",
                    source="sample.txt",
                    fusion_score=0.944,
                )
            ],
        )


def test_ask_question() -> None:
    app.dependency_overrides[get_qa_service] = lambda: FakeQAService()
    client = TestClient(app)

    response = client.post("/qa/ask", json={"question": "What is GraphRAG?", "top_k": 3})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {
        "question": "What is GraphRAG?",
        "answer": "GraphRAG combines retrieval and generation.",
        "sources": [
            {
                "chunk_id": "sample_chunk_0000",
                "document_id": "sample",
                "source": "sample.txt",
                "file_name": "sample.txt",
                "score": 0.91,
                "content": "GraphRAG connects vector search and graph traversal.",
            }
        ],
        "graph_sources": [
            {
                "center_name": "GraphRAG",
                "center_type": "Method",
                "source_name": "GraphRAG",
                "source_type": "Method",
                "relation_type": "SOLVES_TASK",
                "target_name": "question answering",
                "target_type": "Task",
                "chunk_id": "sample_chunk_0000",
                "document_id": "sample",
                "source": "sample.txt",
                "evidence": "GraphRAG improves question answering.",
                "confidence": 0.9,
            }
        ],
        "citations": [
            {
                "evidence_id": "V1+G1",
                "evidence_type": "hybrid",
                "document_id": "sample",
                "chunk_id": "sample_chunk_0000",
                "source": "sample.txt",
                "fusion_score": 0.944,
            }
        ],
    }


def test_ask_question_rejects_invalid_top_k() -> None:
    app.dependency_overrides[get_qa_service] = lambda: FakeQAService()
    client = TestClient(app)

    response = client.post("/qa/ask", json={"question": "GraphRAG", "top_k": 0})

    app.dependency_overrides.clear()
    assert response.status_code == 422


def test_ask_question_rejects_empty_question() -> None:
    app.dependency_overrides[get_qa_service] = lambda: FakeQAService()
    client = TestClient(app)

    response = client.post("/qa/ask", json={"question": "", "top_k": 3})

    app.dependency_overrides.clear()
    assert response.status_code == 422


def test_build_evidence_reranker_uses_keyword_by_default() -> None:
    reranker = build_evidence_reranker(Settings(_env_file=None))

    assert isinstance(reranker, KeywordOverlapEvidenceReranker)


def test_build_evidence_reranker_wraps_bge_with_fallback() -> None:
    reranker = build_evidence_reranker(Settings(reranker_type="bge", _env_file=None))

    assert isinstance(reranker, FallbackEvidenceReranker)
    assert isinstance(reranker.fallback, KeywordOverlapEvidenceReranker)
