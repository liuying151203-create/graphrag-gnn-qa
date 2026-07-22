from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from graphrag_gnn_qa.graph.extractor import GraphExtractor, graph_result_to_dict
from graphrag_gnn_qa.ingestion.document_loader import DocumentLoader
from graphrag_gnn_qa.ingestion.text_splitter import TextSplitter
from graphrag_gnn_qa.vectorstore.embedding import EmbeddingModel
from graphrag_gnn_qa.vectorstore.milvus_client import EmbeddingRecord, infer_embedding_dimension

class IngestionVectorStore(Protocol):
    def document_exists(self, document_id: str) -> bool:
        ...

    def create_collection(self, dimension: int, drop_existing: bool = False) -> None:
        ...

    def upsert_records(self, records: list[EmbeddingRecord]) -> int:
        ...


class IngestionGraphStore(Protocol):
    def upsert_graph_records(self, records: list[dict[str, Any]], create_constraints: bool = True) -> int:
        ...


@dataclass(frozen=True)
class IngestionTimings:
    parse_ms: float
    chunk_ms: float
    embedding_ms: float
    graph_extraction_ms: float
    vector_write_ms: float
    graph_write_ms: float
    total_ms: float


@dataclass(frozen=True)
class DocumentIngestionResult:
    status: Literal["completed"]
    document_id: str
    content_sha256: str
    filename: str
    file_type: str
    chunk_count: int
    embedding_count: int
    entity_count: int
    relation_count: int
    timings: IngestionTimings


class DocumentValidationError(ValueError):
    pass


class UnsupportedDocumentTypeError(DocumentValidationError):
    pass


class DuplicateDocumentError(ValueError):
    def __init__(self, document_id: str, filename: str) -> None:
        self.document_id = document_id
        self.filename = filename
        super().__init__(f"Document already exists: {document_id}")


class IngestionStageError(RuntimeError):
    def __init__(
        self,
        stage: str,
        document_id: str,
        status: Literal["failed", "partial_failed"],
        cause: Exception,
    ) -> None:
        self.stage = stage
        self.document_id = document_id
        self.status = status
        self.cause = cause
        super().__init__(f"Document ingestion failed during {stage}")


class DocumentIngestionService:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: IngestionVectorStore,
        graph_extractor: GraphExtractor,
        graph_store: IngestionGraphStore,
        document_loader: DocumentLoader | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        embedding_batch_size: int = 16,
    ) -> None:
        if embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be greater than 0")
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.graph_extractor = graph_extractor
        self.graph_store = graph_store
        self.document_loader = document_loader or DocumentLoader()
        self.text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embedding_batch_size = embedding_batch_size

    def ingest(self, filename: str, content: bytes) -> DocumentIngestionResult:
        total_started = time.perf_counter()
        safe_filename = normalize_upload_filename(filename)
        if not content:
            raise DocumentValidationError("Uploaded document is empty")
        suffix = Path(safe_filename).suffix.lower()
        if suffix not in self.document_loader.supported_extensions:
            raise UnsupportedDocumentTypeError(f"Unsupported document type: {suffix}")

        content_sha256 = hashlib.sha256(content).hexdigest()
        document_id = build_content_document_id(content_sha256)
        try:
            if self.vector_store.document_exists(document_id):
                raise DuplicateDocumentError(document_id=document_id, filename=safe_filename)
        except DuplicateDocumentError:
            raise
        except Exception as exc:
            raise IngestionStageError("duplicate_check", document_id, "failed", exc) from exc

        parse_started = time.perf_counter()
        try:
            document = self.document_loader.load_bytes(
                content=content,
                file_name=safe_filename,
                source=safe_filename,
            )
        except ValueError as exc:
            raise DocumentValidationError(str(exc)) from exc
        except Exception as exc:
            raise IngestionStageError("parse", document_id, "failed", exc) from exc
        parse_ms = _elapsed_ms(parse_started)

        chunk_started = time.perf_counter()
        chunks = self.text_splitter.split(document.content, document_id=document_id)
        if not chunks:
            raise DocumentValidationError("Document contains no extractable text")
        chunk_records = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": document_id,
                "content": chunk.content,
                "source": document.source,
                "file_name": document.file_name,
                "file_type": document.file_type,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index,
            }
            for chunk in chunks
        ]
        chunk_ms = _elapsed_ms(chunk_started)

        embedding_started = time.perf_counter()
        try:
            embedding_records = self._build_embedding_records(chunk_records)
        except Exception as exc:
            raise IngestionStageError("embedding", document_id, "failed", exc) from exc
        embedding_ms = _elapsed_ms(embedding_started)

        graph_extraction_started = time.perf_counter()
        try:
            graph_records = [
                graph_result_to_dict(self.graph_extractor.extract_from_chunk(chunk))
                for chunk in chunk_records
            ]
        except Exception as exc:
            raise IngestionStageError("graph_extraction", document_id, "failed", exc) from exc
        graph_extraction_ms = _elapsed_ms(graph_extraction_started)

        graph_write_started = time.perf_counter()
        try:
            self.graph_store.upsert_graph_records(graph_records)
        except Exception as exc:
            raise IngestionStageError("graph_write", document_id, "partial_failed", exc) from exc
        graph_write_ms = _elapsed_ms(graph_write_started)

        vector_write_started = time.perf_counter()
        try:
            dimension = infer_embedding_dimension(embedding_records)
            self.vector_store.create_collection(dimension=dimension)
            embedding_count = self.vector_store.upsert_records(embedding_records)
        except Exception as exc:
            raise IngestionStageError("vector_write", document_id, "partial_failed", exc) from exc
        vector_write_ms = _elapsed_ms(vector_write_started)

        entity_count, relation_count = count_graph_items(graph_records)
        return DocumentIngestionResult(
            status="completed",
            document_id=document_id,
            content_sha256=content_sha256,
            filename=safe_filename,
            file_type=document.file_type,
            chunk_count=len(chunk_records),
            embedding_count=embedding_count,
            entity_count=entity_count,
            relation_count=relation_count,
            timings=IngestionTimings(
                parse_ms=parse_ms,
                chunk_ms=chunk_ms,
                embedding_ms=embedding_ms,
                graph_extraction_ms=graph_extraction_ms,
                vector_write_ms=vector_write_ms,
                graph_write_ms=graph_write_ms,
                total_ms=_elapsed_ms(total_started),
            ),
        )

    def _build_embedding_records(self, chunk_records: list[dict[str, Any]]) -> list[EmbeddingRecord]:
        records = []
        for start in range(0, len(chunk_records), self.embedding_batch_size):
            batch = chunk_records[start : start + self.embedding_batch_size]
            embeddings = self.embedding_model.embed_texts([chunk["content"] for chunk in batch])
            if len(embeddings) != len(batch):
                raise ValueError("Embedding model returned an unexpected number of vectors")
            records.extend(
                EmbeddingRecord(
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    content=chunk["content"],
                    source=chunk["source"],
                    file_name=chunk["file_name"],
                    file_type=chunk["file_type"],
                    embedding=[float(value) for value in embedding],
                )
                for chunk, embedding in zip(batch, embeddings)
            )
        return records


def normalize_upload_filename(filename: str) -> str:
    safe_filename = str(filename or "").replace("\\", "/").rsplit("/", maxsplit=1)[-1].strip()
    if not safe_filename or safe_filename in {".", ".."}:
        raise DocumentValidationError("Uploaded document must have a valid filename")
    if len(safe_filename) > 512:
        raise DocumentValidationError("Uploaded document filename must not exceed 512 characters")
    return safe_filename


def build_content_document_id(content_sha256: str) -> str:
    normalized_hash = str(content_sha256).strip().lower()
    if len(normalized_hash) != 64 or any(character not in "0123456789abcdef" for character in normalized_hash):
        raise ValueError("content_sha256 must be a SHA-256 hexadecimal digest")
    return f"doc_{normalized_hash[:32]}"


def count_graph_items(records: list[dict[str, Any]]) -> tuple[int, int]:
    entity_ids = set()
    relation_count = 0
    for record in records:
        for entity in record.get("entities") or []:
            name = " ".join(str(entity.get("name") or "").casefold().split())
            entity_type = str(entity.get("type") or "Concept")
            if name:
                entity_ids.add((entity_type, name))
        relation_count += len(record.get("relations") or [])
    return len(entity_ids), relation_count


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
