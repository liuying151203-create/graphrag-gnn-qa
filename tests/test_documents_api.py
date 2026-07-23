from fastapi.testclient import TestClient

from graphrag_gnn_qa.api.dependencies import get_runtime_resources
from graphrag_gnn_qa.api.routes_documents import (
    get_document_ingestion_service,
    get_document_lifecycle_service,
)
from graphrag_gnn_qa.config import Settings, get_settings
from graphrag_gnn_qa.ingestion.service import (
    DeletionTimings,
    DocumentDeletionResult,
    DocumentDeletionStageError,
    DocumentIdValidationError,
    DocumentIngestionResult,
    DocumentNotFoundError,
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

    def ingest(
        self,
        filename: str,
        content: bytes,
        overwrite: bool = False,
    ) -> DocumentIngestionResult:
        self.calls.append((filename, content, overwrite))
        if self.error is not None:
            raise self.error
        return DocumentIngestionResult(
            status="completed",
            operation="replaced" if overwrite else "created",
            document_id="doc_123",
            content_sha256="a" * 64,
            filename=filename,
            file_type="txt",
            chunk_count=2,
            embedding_count=2,
            entity_count=3,
            relation_count=2,
            deleted_chunk_count=2 if overwrite else 0,
            deleted_relation_count=1 if overwrite else 0,
            deleted_entity_count=0,
            timings=IngestionTimings(
                parse_ms=1.0,
                chunk_ms=2.0,
                embedding_ms=3.0,
                graph_extraction_ms=4.0,
                cleanup_ms=7.0 if overwrite else 0.0,
                vector_write_ms=5.0,
                graph_write_ms=6.0,
                total_ms=28.0 if overwrite else 21.0,
            ),
        )


class FakeLifecycleService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = []

    def delete(
        self,
        document_id: str,
        require_exists: bool = True,
    ) -> DocumentDeletionResult:
        self.calls.append((document_id, require_exists))
        if self.error is not None:
            raise self.error
        return DocumentDeletionResult(
            status="completed",
            document_id=document_id,
            deleted_chunk_count=3,
            deleted_relation_count=2,
            deleted_entity_count=1,
            timings=DeletionTimings(
                graph_delete_ms=2.0,
                vector_delete_ms=3.0,
                total_ms=5.0,
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
    assert service.calls == [("paper.txt", b"GraphRAG content", False)]
    assert response.json() == {
        "status": "completed",
        "operation": "created",
        "document_id": "doc_123",
        "content_sha256": "a" * 64,
        "filename": "paper.txt",
        "file_type": "txt",
        "chunk_count": 2,
        "embedding_count": 2,
        "entity_count": 3,
        "relation_count": 2,
        "deleted_chunk_count": 0,
        "deleted_relation_count": 0,
        "deleted_entity_count": 0,
        "timings": {
            "parse_ms": 1.0,
            "chunk_ms": 2.0,
            "embedding_ms": 3.0,
            "graph_extraction_ms": 4.0,
            "cleanup_ms": 0.0,
            "vector_write_ms": 5.0,
            "graph_write_ms": 6.0,
            "total_ms": 21.0,
        },
    }


def test_upload_document_passes_overwrite_flag() -> None:
    service = FakeIngestionService()
    app.dependency_overrides[get_document_ingestion_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/documents/upload",
        data={"overwrite": "true"},
        files={"file": ("paper.txt", b"GraphRAG content", "text/plain")},
    )

    assert response.status_code == 201
    assert service.calls == [("paper.txt", b"GraphRAG content", True)]
    assert response.json()["operation"] == "replaced"
    assert response.json()["deleted_chunk_count"] == 2
    assert response.json()["timings"]["cleanup_ms"] == 7.0


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
        "deleted_chunk_count": 0,
        "deleted_relation_count": 0,
        "deleted_entity_count": 0,
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


def test_delete_document_returns_cleanup_summary() -> None:
    service = FakeLifecycleService()
    app.dependency_overrides[get_document_lifecycle_service] = lambda: service
    client = TestClient(app)

    response = client.delete("/documents/doc_123")

    assert response.status_code == 200
    assert service.calls == [("doc_123", True)]
    assert response.json() == {
        "status": "completed",
        "document_id": "doc_123",
        "deleted_chunk_count": 3,
        "deleted_relation_count": 2,
        "deleted_entity_count": 1,
        "timings": {
            "graph_delete_ms": 2.0,
            "vector_delete_ms": 3.0,
            "total_ms": 5.0,
        },
    }


def test_delete_document_maps_not_found_and_invalid_id() -> None:
    client = TestClient(app)
    app.dependency_overrides[get_document_lifecycle_service] = lambda: FakeLifecycleService(
        error=DocumentNotFoundError("doc_missing")
    )

    missing_response = client.delete("/documents/doc_missing")

    app.dependency_overrides[get_document_lifecycle_service] = lambda: FakeLifecycleService(
        error=DocumentIdValidationError("invalid document id")
    )
    invalid_response = client.delete("/documents/unsafe")

    assert missing_response.status_code == 404
    assert missing_response.json()["detail"] == {
        "code": "document_not_found",
        "document_id": "doc_missing",
    }
    assert invalid_response.status_code == 422
    assert invalid_response.json()["detail"]["code"] == "invalid_document_id"


def test_delete_document_sanitizes_partial_failure() -> None:
    service = FakeLifecycleService(
        error=DocumentDeletionStageError(
            stage="vector_delete",
            document_id="doc_123",
            status="partial_failed",
            cause=ConnectionError("secret milvus address"),
            deleted_relation_count=2,
            deleted_entity_count=1,
        )
    )
    app.dependency_overrides[get_document_lifecycle_service] = lambda: service
    client = TestClient(app)

    response = client.delete("/documents/doc_123")

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "document_deletion_failed",
        "status": "partial_failed",
        "stage": "vector_delete",
        "document_id": "doc_123",
        "message": "Document deletion failed during vector_delete",
        "deleted_chunk_count": 0,
        "deleted_relation_count": 2,
        "deleted_entity_count": 1,
    }
    assert "secret" not in response.text
