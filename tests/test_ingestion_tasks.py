import threading
import time

import pytest

from graphrag_gnn_qa.ingestion.service import (
    DocumentIngestionResult,
    IngestionProgress,
    IngestionStageError,
    IngestionTimings,
    UnsupportedDocumentTypeError,
)
from graphrag_gnn_qa.ingestion.tasks import (
    IngestionTaskManager,
    IngestionTaskNotFoundError,
    IngestionTaskQueueFullError,
)


class FakeIngestionService:
    def __init__(
        self,
        error: Exception | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.error = error
        self.release = release
        self.started = threading.Event()
        self.calls = []

    def ingest(
        self,
        filename: str,
        content: bytes,
        overwrite: bool = False,
        progress_callback=None,
    ) -> DocumentIngestionResult:
        self.calls.append((filename, content, overwrite))
        self.started.set()
        if progress_callback is not None:
            progress_callback(IngestionProgress(stage="embedding", progress=40))
        if self.release is not None:
            assert self.release.wait(timeout=2)
        if self.error is not None:
            raise self.error
        if progress_callback is not None:
            progress_callback(IngestionProgress(stage="vector_write", progress=92))
        return build_result(filename, overwrite)


def build_result(filename: str, overwrite: bool = False) -> DocumentIngestionResult:
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
        deleted_chunk_count=0,
        deleted_relation_count=0,
        deleted_entity_count=0,
        timings=IngestionTimings(
            parse_ms=1.0,
            chunk_ms=2.0,
            embedding_ms=3.0,
            graph_extraction_ms=4.0,
            cleanup_ms=0.0,
            vector_write_ms=5.0,
            graph_write_ms=6.0,
            total_ms=21.0,
        ),
    )


def wait_for_terminal(manager: IngestionTaskManager, task_id: str):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        snapshot = manager.get(task_id)
        if snapshot.status in {"completed", "failed", "partial_failed"}:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"Task did not finish: {task_id}")


def test_task_manager_runs_ingestion_and_returns_result() -> None:
    service = FakeIngestionService()
    manager = IngestionTaskManager(service)
    try:
        submitted = manager.submit("paper.txt", b"GraphRAG", overwrite=True)
        completed = wait_for_terminal(manager, submitted.task_id)
    finally:
        manager.close()

    assert submitted.status == "pending"
    assert submitted.progress == 0
    assert submitted.stage == "queued"
    assert completed.status == "completed"
    assert completed.progress == 100
    assert completed.stage == "completed"
    assert completed.result is not None
    assert completed.result.operation == "replaced"
    assert completed.error is None
    assert completed.started_at is not None
    assert completed.completed_at is not None
    assert service.calls == [("paper.txt", b"GraphRAG", True)]


def test_task_manager_keeps_queued_task_pending_with_single_worker() -> None:
    release = threading.Event()
    service = FakeIngestionService(release=release)
    manager = IngestionTaskManager(service, max_workers=1)
    try:
        first = manager.submit("first.txt", b"first")
        assert service.started.wait(timeout=1)
        second = manager.submit("second.txt", b"second")

        queued = manager.get(second.task_id)

        assert queued.status == "pending"
        assert queued.stage == "queued"
        release.set()
        assert wait_for_terminal(manager, first.task_id).status == "completed"
        assert wait_for_terminal(manager, second.task_id).status == "completed"
    finally:
        release.set()
        manager.close()


def test_task_manager_rejects_submission_when_queue_is_full() -> None:
    release = threading.Event()
    service = FakeIngestionService(release=release)
    manager = IngestionTaskManager(service, queue_limit=1)
    try:
        manager.submit("first.txt", b"first")
        assert service.started.wait(timeout=1)

        with pytest.raises(IngestionTaskQueueFullError) as exc_info:
            manager.submit("second.txt", b"second")

        assert exc_info.value.queue_limit == 1
    finally:
        release.set()
        manager.close()


def test_task_manager_preserves_partial_failure_details() -> None:
    service = FakeIngestionService(
        error=IngestionStageError(
            stage="vector_write",
            document_id="doc_123",
            status="partial_failed",
            cause=ConnectionError("secret address"),
            deleted_relation_count=2,
        )
    )
    manager = IngestionTaskManager(service)
    try:
        submitted = manager.submit("paper.txt", b"GraphRAG")
        failed = wait_for_terminal(manager, submitted.task_id)
    finally:
        manager.close()

    assert failed.status == "partial_failed"
    assert failed.stage == "vector_write"
    assert failed.error is not None
    assert failed.error.code == "ingestion_stage_failed"
    assert failed.error.deleted_relation_count == 2
    assert "secret" not in failed.error.message


def test_task_manager_validates_submission_and_prunes_history() -> None:
    manager = IngestionTaskManager(FakeIngestionService(), history_limit=1)
    try:
        with pytest.raises(UnsupportedDocumentTypeError):
            manager.submit("paper.docx", b"content")

        first = manager.submit("first.txt", b"first")
        assert wait_for_terminal(manager, first.task_id).status == "completed"
        second = manager.submit("second.txt", b"second")

        with pytest.raises(IngestionTaskNotFoundError):
            manager.get(first.task_id)
        assert wait_for_terminal(manager, second.task_id).status == "completed"
    finally:
        manager.close()
