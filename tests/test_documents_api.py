from datetime import datetime, timezone

from fastapi.testclient import TestClient

from graphrag_gnn_qa.api.dependencies import get_runtime_resources
from graphrag_gnn_qa.api.routes_documents import (
    get_document_ingestion_service,
    get_document_lifecycle_service,
    get_ingestion_task_manager,
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
from graphrag_gnn_qa.ingestion.tasks import (
    IngestionTaskError,
    IngestionTaskNotFoundError,
    IngestionTaskQueueFullError,
    IngestionTaskSnapshot,
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


class FakeTaskManager:
    def __init__(
        self,
        task: IngestionTaskSnapshot,
        get_error: Exception | None = None,
        submit_error: Exception | None = None,
    ) -> None:
        self.task = task
        self.get_error = get_error
        self.submit_error = submit_error
        self.submit_calls = []
        self.get_calls = []

    def submit(
        self,
        filename: str,
        content: bytes,
        overwrite: bool = False,
    ) -> IngestionTaskSnapshot:
        self.submit_calls.append((filename, content, overwrite))
        if self.submit_error is not None:
            raise self.submit_error
        return self.task

    def get(self, task_id: str) -> IngestionTaskSnapshot:
        self.get_calls.append(task_id)
        if self.get_error is not None:
            raise self.get_error
        return self.task


def build_task_snapshot(
    status: str = "pending",
    result: DocumentIngestionResult | None = None,
    error: IngestionTaskError | None = None,
) -> IngestionTaskSnapshot:
    now = datetime(2026, 7, 28, 1, 2, 3, tzinfo=timezone.utc)
    terminal = status in {"completed", "failed", "partial_failed"}
    return IngestionTaskSnapshot(
        task_id="ing_123",
        status=status,
        progress=100 if status == "completed" else 40 if terminal else 0,
        stage="completed" if status == "completed" else error.stage if error else "queued",
        filename="paper.txt",
        overwrite=False,
        document_id="doc_123",
        created_at=now,
        started_at=now if status != "pending" else None,
        updated_at=now,
        completed_at=now if terminal else None,
        result=result,
        error=error,
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


def test_queue_document_upload_returns_pending_task_and_location() -> None:
    task_manager = FakeTaskManager(build_task_snapshot())
    app.dependency_overrides[get_ingestion_task_manager] = lambda: task_manager
    client = TestClient(app)

    response = client.post(
        "/documents/upload/async",
        data={"overwrite": "true"},
        files={"file": ("paper.txt", b"GraphRAG content", "text/plain")},
    )

    assert response.status_code == 202
    assert response.headers["location"] == "/documents/tasks/ing_123"
    assert task_manager.submit_calls == [("paper.txt", b"GraphRAG content", True)]
    assert response.json()["task_id"] == "ing_123"
    assert response.json()["status"] == "pending"
    assert response.json()["progress"] == 0
    assert response.json()["result"] is None
    assert response.json()["error"] is None


def test_get_document_ingestion_task_returns_completed_result() -> None:
    result = FakeIngestionService().ingest("paper.txt", b"content")
    task_manager = FakeTaskManager(
        build_task_snapshot(status="completed", result=result)
    )
    app.dependency_overrides[get_ingestion_task_manager] = lambda: task_manager
    client = TestClient(app)

    response = client.get("/documents/tasks/ing_123")

    assert response.status_code == 200
    assert task_manager.get_calls == ["ing_123"]
    assert response.json()["status"] == "completed"
    assert response.json()["progress"] == 100
    assert response.json()["result"]["operation"] == "created"
    assert response.json()["result"]["chunk_count"] == 2


def test_queue_document_upload_returns_retryable_queue_full_error() -> None:
    task_manager = FakeTaskManager(
        build_task_snapshot(),
        submit_error=IngestionTaskQueueFullError(queue_limit=10),
    )
    app.dependency_overrides[get_ingestion_task_manager] = lambda: task_manager
    client = TestClient(app)

    response = client.post(
        "/documents/upload/async",
        files={"file": ("paper.txt", b"GraphRAG content", "text/plain")},
    )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "5"
    assert response.json()["detail"] == {
        "code": "ingestion_queue_full",
        "queue_limit": 10,
    }


def test_get_document_ingestion_task_returns_partial_failure() -> None:
    error = IngestionTaskError(
        code="ingestion_stage_failed",
        message="Document ingestion failed during vector_write",
        stage="vector_write",
        deleted_chunk_count=2,
    )
    task_manager = FakeTaskManager(
        build_task_snapshot(status="partial_failed", error=error)
    )
    app.dependency_overrides[get_ingestion_task_manager] = lambda: task_manager
    client = TestClient(app)

    response = client.get("/documents/tasks/ing_123")

    assert response.status_code == 200
    assert response.json()["status"] == "partial_failed"
    assert response.json()["error"] == {
        "code": "ingestion_stage_failed",
        "message": "Document ingestion failed during vector_write",
        "stage": "vector_write",
        "deleted_chunk_count": 2,
        "deleted_relation_count": 0,
        "deleted_entity_count": 0,
    }


def test_get_document_ingestion_task_maps_not_found() -> None:
    task_manager = FakeTaskManager(
        build_task_snapshot(),
        get_error=IngestionTaskNotFoundError("ing_missing"),
    )
    app.dependency_overrides[get_ingestion_task_manager] = lambda: task_manager
    client = TestClient(app)

    response = client.get("/documents/tasks/ing_missing")

    assert response.status_code == 404
    assert response.json()["detail"] == {
        "code": "ingestion_task_not_found",
        "task_id": "ing_missing",
    }


def test_background_ingestion_is_unavailable_without_task_manager() -> None:
    resources = type("Resources", (), {"ingestion_task_manager": None})()
    app.dependency_overrides[get_runtime_resources] = lambda: resources
    client = TestClient(app)

    response = client.get("/documents/tasks/ing_123")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "LLM_API_KEY is not configured for background document ingestion"
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
