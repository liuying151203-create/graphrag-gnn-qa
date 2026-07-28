from __future__ import annotations

import hashlib
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from graphrag_gnn_qa.ingestion.document_loader import DocumentLoader
from graphrag_gnn_qa.ingestion.service import (
    DocumentIngestionResult,
    DocumentValidationError,
    DuplicateDocumentError,
    IngestionProgress,
    IngestionProgressCallback,
    IngestionStageError,
    UnsupportedDocumentTypeError,
    build_content_document_id,
    normalize_upload_filename,
)

IngestionTaskStatus = Literal[
    "pending",
    "processing",
    "completed",
    "failed",
    "partial_failed",
]
TERMINAL_TASK_STATUSES = {"completed", "failed", "partial_failed"}


class BackgroundIngestionService(Protocol):
    def ingest(
        self,
        filename: str,
        content: bytes,
        overwrite: bool = False,
        progress_callback: IngestionProgressCallback | None = None,
    ) -> DocumentIngestionResult:
        ...


@dataclass(frozen=True)
class IngestionTaskError:
    code: str
    message: str
    stage: str
    deleted_chunk_count: int = 0
    deleted_relation_count: int = 0
    deleted_entity_count: int = 0


@dataclass(frozen=True)
class IngestionTaskSnapshot:
    task_id: str
    status: IngestionTaskStatus
    progress: int
    stage: str
    filename: str
    overwrite: bool
    document_id: str
    created_at: datetime
    started_at: datetime | None
    updated_at: datetime
    completed_at: datetime | None
    result: DocumentIngestionResult | None
    error: IngestionTaskError | None


@dataclass
class _IngestionTaskState:
    task_id: str
    status: IngestionTaskStatus
    progress: int
    stage: str
    filename: str
    overwrite: bool
    document_id: str
    created_at: datetime
    started_at: datetime | None
    updated_at: datetime
    completed_at: datetime | None
    result: DocumentIngestionResult | None
    error: IngestionTaskError | None

    def snapshot(self) -> IngestionTaskSnapshot:
        return IngestionTaskSnapshot(**self.__dict__)


class IngestionTaskNotFoundError(LookupError):
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        super().__init__(f"Ingestion task not found: {task_id}")


class IngestionTaskQueueFullError(RuntimeError):
    def __init__(self, queue_limit: int) -> None:
        self.queue_limit = queue_limit
        super().__init__(f"Ingestion task queue is full: {queue_limit}")


class IngestionTaskManager:
    def __init__(
        self,
        ingestion_service: BackgroundIngestionService,
        max_workers: int = 1,
        history_limit: int = 100,
        queue_limit: int = 10,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
        if history_limit <= 0:
            raise ValueError("history_limit must be greater than 0")
        if queue_limit <= 0:
            raise ValueError("queue_limit must be greater than 0")
        self.ingestion_service = ingestion_service
        self.history_limit = history_limit
        self.queue_limit = queue_limit
        self._tasks: dict[str, _IngestionTaskState] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="document-ingestion",
        )
        self._closed = False

    def submit(
        self,
        filename: str,
        content: bytes,
        overwrite: bool = False,
    ) -> IngestionTaskSnapshot:
        safe_filename = _validate_submission(filename, content)
        content_sha256 = hashlib.sha256(content).hexdigest()
        document_id = build_content_document_id(content_sha256)
        task_id = f"ing_{uuid.uuid4().hex}"
        now = _utcnow()
        state = _IngestionTaskState(
            task_id=task_id,
            status="pending",
            progress=0,
            stage="queued",
            filename=safe_filename,
            overwrite=overwrite,
            document_id=document_id,
            created_at=now,
            started_at=None,
            updated_at=now,
            completed_at=None,
            result=None,
            error=None,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("Ingestion task manager is closed")
            active_task_count = sum(
                state.status not in TERMINAL_TASK_STATUSES
                for state in self._tasks.values()
            )
            if active_task_count >= self.queue_limit:
                raise IngestionTaskQueueFullError(self.queue_limit)
            self._prune_terminal_tasks()
            self._tasks[task_id] = state
            snapshot = state.snapshot()
        try:
            self._executor.submit(
                self._run_task,
                task_id,
                safe_filename,
                content,
                overwrite,
            )
        except Exception:
            with self._lock:
                self._tasks.pop(task_id, None)
            raise
        return snapshot

    def get(self, task_id: str) -> IngestionTaskSnapshot:
        normalized_task_id = str(task_id or "").strip()
        with self._lock:
            state = self._tasks.get(normalized_task_id)
            if state is None:
                raise IngestionTaskNotFoundError(normalized_task_id)
            return state.snapshot()

    def close(self, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _run_task(
        self,
        task_id: str,
        filename: str,
        content: bytes,
        overwrite: bool,
    ) -> None:
        now = _utcnow()
        with self._lock:
            state = self._tasks[task_id]
            state.status = "processing"
            state.progress = 1
            state.stage = "validation"
            state.started_at = now
            state.updated_at = now

        try:
            result = self.ingestion_service.ingest(
                filename=filename,
                content=content,
                overwrite=overwrite,
                progress_callback=lambda progress: self._update_progress(task_id, progress),
            )
        except DuplicateDocumentError:
            self._finish_error(
                task_id=task_id,
                status="failed",
                error=IngestionTaskError(
                    code="duplicate_document",
                    message="Document content already exists",
                    stage="duplicate_check",
                ),
            )
        except UnsupportedDocumentTypeError as exc:
            self._finish_error(
                task_id=task_id,
                status="failed",
                error=IngestionTaskError(
                    code="unsupported_document_type",
                    message=str(exc),
                    stage="validation",
                ),
            )
        except DocumentValidationError as exc:
            self._finish_error(
                task_id=task_id,
                status="failed",
                error=IngestionTaskError(
                    code="invalid_document",
                    message=str(exc),
                    stage="validation",
                ),
            )
        except IngestionStageError as exc:
            self._finish_error(
                task_id=task_id,
                status=exc.status,
                error=IngestionTaskError(
                    code="ingestion_stage_failed",
                    message=str(exc),
                    stage=exc.stage,
                    deleted_chunk_count=exc.deleted_chunk_count,
                    deleted_relation_count=exc.deleted_relation_count,
                    deleted_entity_count=exc.deleted_entity_count,
                ),
            )
        except Exception:
            self._finish_error(
                task_id=task_id,
                status="failed",
                error=IngestionTaskError(
                    code="internal_error",
                    message="Document ingestion task failed",
                    stage=self._current_stage(task_id),
                ),
            )
        else:
            completed_at = _utcnow()
            with self._lock:
                state = self._tasks[task_id]
                state.status = "completed"
                state.progress = 100
                state.stage = "completed"
                state.result = result
                state.updated_at = completed_at
                state.completed_at = completed_at

    def _update_progress(self, task_id: str, progress: IngestionProgress) -> None:
        if progress.progress >= 100:
            return
        with self._lock:
            state = self._tasks.get(task_id)
            if state is None or state.status != "processing":
                return
            state.progress = max(state.progress, min(progress.progress, 99))
            state.stage = progress.stage
            state.updated_at = _utcnow()

    def _finish_error(
        self,
        task_id: str,
        status: Literal["failed", "partial_failed"],
        error: IngestionTaskError,
    ) -> None:
        completed_at = _utcnow()
        with self._lock:
            state = self._tasks[task_id]
            state.status = status
            state.stage = error.stage
            state.error = error
            state.updated_at = completed_at
            state.completed_at = completed_at

    def _current_stage(self, task_id: str) -> str:
        with self._lock:
            state = self._tasks.get(task_id)
            return state.stage if state is not None else "unknown"

    def _prune_terminal_tasks(self) -> None:
        overflow = len(self._tasks) - self.history_limit + 1
        if overflow <= 0:
            return
        terminal_tasks = sorted(
            (
                state
                for state in self._tasks.values()
                if state.status in TERMINAL_TASK_STATUSES
            ),
            key=lambda state: state.completed_at or state.created_at,
        )
        for state in terminal_tasks[:overflow]:
            self._tasks.pop(state.task_id, None)


def _validate_submission(filename: str, content: bytes) -> str:
    safe_filename = normalize_upload_filename(filename)
    if not content:
        raise DocumentValidationError("Uploaded document is empty")
    suffix = Path(safe_filename).suffix.lower()
    if suffix not in DocumentLoader.supported_extensions:
        raise UnsupportedDocumentTypeError(f"Unsupported document type: {suffix}")
    return safe_filename


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
