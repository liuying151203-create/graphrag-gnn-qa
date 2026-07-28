from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from graphrag_gnn_qa.api.dependencies import get_runtime_resources
from graphrag_gnn_qa.config import Settings, get_settings
from graphrag_gnn_qa.ingestion.service import (
    DocumentDeletionResult,
    DocumentDeletionStageError,
    DocumentIdValidationError,
    DocumentIngestionResult,
    DocumentNotFoundError,
    DocumentValidationError,
    DuplicateDocumentError,
    IngestionStageError,
    UnsupportedDocumentTypeError,
)
from graphrag_gnn_qa.ingestion.tasks import (
    IngestionTaskError,
    IngestionTaskNotFoundError,
    IngestionTaskQueueFullError,
    IngestionTaskSnapshot,
)
from graphrag_gnn_qa.runtime import RuntimeResources


class IngestionTimingsResponse(BaseModel):
    parse_ms: float
    chunk_ms: float
    embedding_ms: float
    graph_extraction_ms: float
    cleanup_ms: float
    vector_write_ms: float
    graph_write_ms: float
    total_ms: float


class DocumentUploadResponse(BaseModel):
    status: Literal["completed"]
    operation: Literal["created", "replaced"]
    document_id: str
    content_sha256: str
    filename: str
    file_type: str
    chunk_count: int
    embedding_count: int
    entity_count: int
    relation_count: int
    deleted_chunk_count: int
    deleted_relation_count: int
    deleted_entity_count: int
    timings: IngestionTimingsResponse


class IngestionTaskErrorResponse(BaseModel):
    code: str
    message: str
    stage: str
    deleted_chunk_count: int
    deleted_relation_count: int
    deleted_entity_count: int


class DocumentIngestionTaskResponse(BaseModel):
    task_id: str
    status: Literal[
        "pending",
        "processing",
        "completed",
        "failed",
        "partial_failed",
    ]
    progress: int
    stage: str
    filename: str
    overwrite: bool
    document_id: str
    created_at: datetime
    started_at: datetime | None
    updated_at: datetime
    completed_at: datetime | None
    result: DocumentUploadResponse | None
    error: IngestionTaskErrorResponse | None


class IngestionService(Protocol):
    def ingest(
        self,
        filename: str,
        content: bytes,
        overwrite: bool = False,
    ) -> DocumentIngestionResult:
        ...


class DocumentLifecycle(Protocol):
    def delete(
        self,
        document_id: str,
        require_exists: bool = True,
    ) -> DocumentDeletionResult:
        ...


class BackgroundIngestionTasks(Protocol):
    def submit(
        self,
        filename: str,
        content: bytes,
        overwrite: bool = False,
    ) -> IngestionTaskSnapshot:
        ...

    def get(self, task_id: str) -> IngestionTaskSnapshot:
        ...


class DeletionTimingsResponse(BaseModel):
    graph_delete_ms: float
    vector_delete_ms: float
    total_ms: float


class DocumentDeleteResponse(BaseModel):
    status: Literal["completed"]
    document_id: str
    deleted_chunk_count: int
    deleted_relation_count: int
    deleted_entity_count: int
    timings: DeletionTimingsResponse


router = APIRouter(prefix="/documents", tags=["documents"])


def get_document_ingestion_service(
    resources: RuntimeResources = Depends(get_runtime_resources),
) -> IngestionService:
    ingestion_service = getattr(resources, "ingestion_service", None)
    if ingestion_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM_API_KEY is not configured for document graph extraction",
        )
    return ingestion_service


def get_document_lifecycle_service(
    resources: RuntimeResources = Depends(get_runtime_resources),
) -> DocumentLifecycle:
    lifecycle_service = getattr(resources, "document_lifecycle_service", None)
    if lifecycle_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document lifecycle service is not initialized",
        )
    return lifecycle_service


def get_ingestion_task_manager(
    resources: RuntimeResources = Depends(get_runtime_resources),
) -> BackgroundIngestionTasks:
    task_manager = getattr(resources, "ingestion_task_manager", None)
    if task_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM_API_KEY is not configured for background document ingestion",
        )
    return task_manager


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_409_CONFLICT: {"description": "Document content already exists"},
        status.HTTP_413_CONTENT_TOO_LARGE: {"description": "Document exceeds the configured size limit"},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"description": "Unsupported document extension"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Invalid or unreadable document"},
        status.HTTP_502_BAD_GATEWAY: {"description": "Upstream LLM request failed"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Ingestion dependency is unavailable"},
    },
)
async def upload_document(
    file: UploadFile = File(...),
    overwrite: bool = Form(default=False),
    settings: Settings = Depends(get_settings),
    ingestion_service: IngestionService = Depends(get_document_ingestion_service),
) -> DocumentUploadResponse:
    filename, content = await _read_upload(file, settings.document_upload_max_bytes)

    try:
        result = await run_in_threadpool(
            ingestion_service.ingest,
            filename,
            content,
            overwrite,
        )
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"code": "unsupported_document_type", "message": str(exc)},
        ) from exc
    except DuplicateDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "duplicate_document",
                "document_id": exc.document_id,
                "filename": exc.filename,
            },
        ) from exc
    except DocumentValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": "invalid_document", "message": str(exc)},
        ) from exc
    except IngestionStageError as exc:
        raise HTTPException(
            status_code=_stage_error_status(exc),
            detail={
                "code": "ingestion_stage_failed",
                "status": exc.status,
                "stage": exc.stage,
                "document_id": exc.document_id,
                "message": str(exc),
                "deleted_chunk_count": exc.deleted_chunk_count,
                "deleted_relation_count": exc.deleted_relation_count,
                "deleted_entity_count": exc.deleted_entity_count,
            },
        ) from exc

    return _upload_response(result)


@router.post(
    "/upload/async",
    response_model=DocumentIngestionTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_413_CONTENT_TOO_LARGE: {"description": "Document exceeds the configured size limit"},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"description": "Unsupported document extension"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Invalid document submission"},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Background ingestion queue is full"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Background ingestion is unavailable"},
    },
)
async def queue_document_upload(
    response: Response,
    file: UploadFile = File(...),
    overwrite: bool = Form(default=False),
    settings: Settings = Depends(get_settings),
    task_manager: BackgroundIngestionTasks = Depends(get_ingestion_task_manager),
) -> DocumentIngestionTaskResponse:
    filename, content = await _read_upload(file, settings.document_upload_max_bytes)
    try:
        task = task_manager.submit(
            filename=filename,
            content=content,
            overwrite=overwrite,
        )
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"code": "unsupported_document_type", "message": str(exc)},
        ) from exc
    except DocumentValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": "invalid_document", "message": str(exc)},
        ) from exc
    except IngestionTaskQueueFullError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "ingestion_queue_full",
                "queue_limit": exc.queue_limit,
            },
            headers={"Retry-After": "5"},
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "task_manager_unavailable", "message": str(exc)},
        ) from exc
    response.headers["Location"] = f"/documents/tasks/{task.task_id}"
    return _task_response(task)


@router.get(
    "/tasks/{task_id}",
    response_model=DocumentIngestionTaskResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Ingestion task does not exist"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Background ingestion is unavailable"},
    },
)
async def get_document_ingestion_task(
    task_id: str,
    task_manager: BackgroundIngestionTasks = Depends(get_ingestion_task_manager),
) -> DocumentIngestionTaskResponse:
    try:
        task = task_manager.get(task_id)
    except IngestionTaskNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "ingestion_task_not_found", "task_id": exc.task_id},
        ) from exc
    return _task_response(task)


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Document does not exist"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Invalid document ID"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Deletion dependency is unavailable"},
    },
)
async def delete_document(
    document_id: str,
    lifecycle_service: DocumentLifecycle = Depends(get_document_lifecycle_service),
) -> DocumentDeleteResponse:
    try:
        result = await run_in_threadpool(
            lifecycle_service.delete,
            document_id,
            True,
        )
    except DocumentIdValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": "invalid_document_id", "message": str(exc)},
        ) from exc
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "document_not_found", "document_id": exc.document_id},
        ) from exc
    except DocumentDeletionStageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "document_deletion_failed",
                "status": exc.status,
                "stage": exc.stage,
                "document_id": exc.document_id,
                "message": str(exc),
                "deleted_chunk_count": exc.deleted_chunk_count,
                "deleted_relation_count": exc.deleted_relation_count,
                "deleted_entity_count": exc.deleted_entity_count,
            },
        ) from exc
    return DocumentDeleteResponse(
        status=result.status,
        document_id=result.document_id,
        deleted_chunk_count=result.deleted_chunk_count,
        deleted_relation_count=result.deleted_relation_count,
        deleted_entity_count=result.deleted_entity_count,
        timings=DeletionTimingsResponse(**result.timings.__dict__),
    )


def _stage_error_status(error: IngestionStageError) -> int:
    if isinstance(error.cause, httpx.HTTPError):
        return status.HTTP_502_BAD_GATEWAY
    if error.stage in {
        "duplicate_check",
        "graph_delete",
        "vector_delete",
        "vector_write",
        "graph_write",
    }:
        return status.HTTP_503_SERVICE_UNAVAILABLE
    return status.HTTP_500_INTERNAL_SERVER_ERROR


async def _read_upload(file: UploadFile, max_bytes: int) -> tuple[str, bytes]:
    filename = file.filename or ""
    try:
        content = await file.read(max_bytes + 1)
    finally:
        await file.close()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={
                "code": "document_too_large",
                "max_bytes": max_bytes,
            },
        )
    return filename, content


def _upload_response(result: DocumentIngestionResult) -> DocumentUploadResponse:
    return DocumentUploadResponse(
        status=result.status,
        operation=result.operation,
        document_id=result.document_id,
        content_sha256=result.content_sha256,
        filename=result.filename,
        file_type=result.file_type,
        chunk_count=result.chunk_count,
        embedding_count=result.embedding_count,
        entity_count=result.entity_count,
        relation_count=result.relation_count,
        deleted_chunk_count=result.deleted_chunk_count,
        deleted_relation_count=result.deleted_relation_count,
        deleted_entity_count=result.deleted_entity_count,
        timings=IngestionTimingsResponse(**result.timings.__dict__),
    )


def _task_response(task: IngestionTaskSnapshot) -> DocumentIngestionTaskResponse:
    return DocumentIngestionTaskResponse(
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        stage=task.stage,
        filename=task.filename,
        overwrite=task.overwrite,
        document_id=task.document_id,
        created_at=task.created_at,
        started_at=task.started_at,
        updated_at=task.updated_at,
        completed_at=task.completed_at,
        result=_upload_response(task.result) if task.result is not None else None,
        error=_task_error_response(task.error) if task.error is not None else None,
    )


def _task_error_response(error: IngestionTaskError) -> IngestionTaskErrorResponse:
    return IngestionTaskErrorResponse(**error.__dict__)
