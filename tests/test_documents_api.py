from fastapi.testclient import TestClient

from graphrag_gnn_qa.api.dependencies import get_runtime_resources
from graphrag_gnn_qa.api.routes_documents import get_document_ingestion_service
from graphrag_gnn_qa.config import Settings, get_settings
from graphrag_gnn_qa.ingestion.service import (
    DocumentIngestionResult,
    DuplicateDocumentError,
    IngestionStageError,
    IngestionTimings,
    UnsupportedDocumentTypeError,
)
from graphrag_gnn_qa.main import app


class FakeIngestionService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = []

    def ingest(self, filename: str, content: bytes) -> DocumentIngestionResult:
        self.calls.append((filename, content))
        if self.error is not None:
            raise self.error
        return DocumentIngestionResult(
            status="completed",
            document_id="doc_123",
            content_sha256="a" * 64,
            filename=filename,
            file_type="txt",
            chunk_count=2,
            embedding_count=2,
            entity_count=3,
            relation_count=2,
            timings=IngestionTimings(
                parse_ms=1.0,
                chunk_ms=2.0,
                embedding_ms=3.0,
                graph_extraction_ms=4.0,
                vector_write_ms=5.0,
                graph_write_ms=6.0,
                total_ms=21.0,
            ),
        )


def teardown_function() -> None:
    app.dependency_overrides.clear()


def test_upload_document_returns_ingestion_summary() -> None:
    service = FakeIngestionService()
    app.dependency_overrides[get_document_ingestion_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/documents/upload",
        files={"file": ("paper.txt", b"GraphRAG content", "text/plain")},
    )

    assert response.status_code == 201
    assert service.calls == [("paper.txt", b"GraphRAG content")]
    assert response.json() == {
        "status": "completed",
        "document_id": "doc_123",
        "content_sha256": "a" * 64,
        "filename": "paper.txt",
        "file_type": "txt",
        "chunk_count": 2,
        "embedding_count": 2,
        "entity_count": 3,
        "relation_count": 2,
        "timings": {
            "parse_ms": 1.0,
            "chunk_ms": 2.0,
            "embedding_ms": 3.0,
            "graph_extraction_ms": 4.0,
            "vector_write_ms": 5.0,
            "graph_write_ms": 6.0,
            "total_ms": 21.0,
        },
    }


def test_upload_document_rejects_duplicate() -> None:
    service = FakeIngestionService(
        error=DuplicateDocumentError(document_id="doc_duplicate", filename="paper.txt")
    )
    app.dependency_overrides[get_document_ingestion_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/documents/upload",
        files={"file": ("paper.txt", b"duplicate", "text/plain")},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "duplicate_document",
        "document_id": "doc_duplicate",
        "filename": "paper.txt",
    }


def test_upload_document_rejects_unsupported_type() -> None:
    service = FakeIngestionService(error=UnsupportedDocumentTypeError("Unsupported document type: .docx"))
    app.dependency_overrides[get_document_ingestion_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/documents/upload",
        files={"file": ("paper.docx", b"content", "application/octet-stream")},
    )

    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "unsupported_document_type"


def test_upload_document_enforces_size_limit_before_ingestion() -> None:
    service = FakeIngestionService()
    app.dependency_overrides[get_document_ingestion_service] = lambda: service
    app.dependency_overrides[get_settings] = lambda: Settings(
        document_upload_max_bytes=4,
        _env_file=None,
    )
    client = TestClient(app)

    response = client.post(
        "/documents/upload",
        files={"file": ("paper.txt", b"12345", "text/plain")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == {
        "code": "document_too_large",
        "max_bytes": 4,
    }
    assert service.calls == []


def test_upload_document_sanitizes_stage_failure() -> None:
    service = FakeIngestionService(
        error=IngestionStageError(
            stage="graph_write",
            document_id="doc_123",
            status="partial_failed",
            cause=ConnectionError("secret neo4j address"),
        )
    )
    app.dependency_overrides[get_document_ingestion_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/documents/upload",
        files={"file": ("paper.txt", b"content", "text/plain")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "ingestion_stage_failed",
        "status": "partial_failed",
        "stage": "graph_write",
        "document_id": "doc_123",
        "message": "Document ingestion failed during graph_write",
    }
    assert "secret" not in response.text


def test_upload_document_is_unavailable_without_ingestion_service() -> None:
    resources = type("Resources", (), {"ingestion_service": None})()
    app.dependency_overrides[get_runtime_resources] = lambda: resources
    client = TestClient(app)

    response = client.post(
        "/documents/upload",
        files={"file": ("paper.txt", b"content", "text/plain")},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "LLM_API_KEY is not configured for document graph extraction"
    }
