from __future__ import annotations

from typing import Literal, Protocol

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
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
    filename = file.filename or ""
    try:
        content = await file.read(settings.document_upload_max_bytes + 1)
    finally:
        await file.close()

    if len(content) > settings.document_upload_max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={
                "code": "document_too_large",
                "max_bytes": settings.document_upload_max_bytes,
            },
        )

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
