import hashlib

import pytest

from graphrag_gnn_qa.graph.extractor import (
    GraphEntity,
    GraphExtractionResult,
    GraphRelation,
)
from graphrag_gnn_qa.graph.neo4j_store import GraphDocumentDeletion
from graphrag_gnn_qa.ingestion.service import (
    DocumentDeletionStageError,
    DocumentIdValidationError,
    DocumentIngestionService,
    DocumentLifecycleService,
    DocumentNotFoundError,
    DocumentValidationError,
    DuplicateDocumentError,
    IngestionStageError,
    UnsupportedDocumentTypeError,
    build_content_document_id,
)


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.batch_sizes = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return [[float(len(text)), 1.0, 0.5] for text in texts]


class FakeVectorStore:
    def __init__(
        self,
        document_exists: bool = False,
        delete_count: int = 0,
        delete_error: Exception | None = None,
    ) -> None:
        self.existing_document = document_exists
        self.delete_count = delete_count
        self.delete_error = delete_error
        self.checked_document_ids = []
        self.deleted_document_ids = []
        self.created_dimensions = []
        self.upserted_records = []

    def document_exists(self, document_id: str) -> bool:
        self.checked_document_ids.append(document_id)
        return self.existing_document

    def create_collection(self, dimension: int, drop_existing: bool = False) -> None:
        self.created_dimensions.append(dimension)

    def upsert_records(self, records: list) -> int:
        self.upserted_records.extend(records)
        return len(records)

    def delete_document(self, document_id: str) -> int:
        self.deleted_document_ids.append(document_id)
        if self.delete_error is not None:
            raise self.delete_error
        return self.delete_count


class FakeGraphExtractor:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.chunk_ids = []

    def extract_from_chunk(self, chunk: dict) -> GraphExtractionResult:
        if self.error is not None:
            raise self.error
        self.chunk_ids.append(chunk["chunk_id"])
        return GraphExtractionResult(
            chunk_id=chunk["chunk_id"],
            document_id=chunk["document_id"],
            source=chunk["source"],
            entities=[
                GraphEntity(name="GraphRAG", type="Method"),
                GraphEntity(name="Question Answering", type="Task"),
            ],
            relations=[
                GraphRelation(
                    source_entity="GraphRAG",
                    source_type="Method",
                    relation_type="SOLVES_TASK",
                    target_entity="Question Answering",
                    target_type="Task",
                    evidence=chunk["content"],
                    confidence=0.9,
                )
            ],
        )


class FakeGraphStore:
    def __init__(
        self,
        error: Exception | None = None,
        deletion: GraphDocumentDeletion | None = None,
        delete_error: Exception | None = None,
    ) -> None:
        self.error = error
        self.deletion = deletion or GraphDocumentDeletion(0, 0)
        self.delete_error = delete_error
        self.records = []
        self.deleted_document_ids = []

    def upsert_graph_records(self, records: list[dict], create_constraints: bool = True) -> int:
        if self.error is not None:
            raise self.error
        self.records.extend(records)
        return len(records)

    def delete_document(self, document_id: str) -> GraphDocumentDeletion:
        self.deleted_document_ids.append(document_id)
        if self.delete_error is not None:
            raise self.delete_error
        return self.deletion


def build_service(
    vector_store: FakeVectorStore | None = None,
    graph_extractor: FakeGraphExtractor | None = None,
    graph_store: FakeGraphStore | None = None,
) -> tuple[DocumentIngestionService, FakeEmbeddingModel, FakeVectorStore, FakeGraphStore]:
    embedding_model = FakeEmbeddingModel()
    resolved_vector_store = vector_store or FakeVectorStore()
    resolved_graph_store = graph_store or FakeGraphStore()
    service = DocumentIngestionService(
        embedding_model=embedding_model,
        vector_store=resolved_vector_store,
        graph_extractor=graph_extractor or FakeGraphExtractor(),
        graph_store=resolved_graph_store,
        chunk_size=10,
        chunk_overlap=2,
        embedding_batch_size=2,
    )
    return service, embedding_model, resolved_vector_store, resolved_graph_store


def test_ingest_document_builds_stable_id_and_writes_both_stores() -> None:
    service, embedding_model, vector_store, graph_store = build_service()
    content = b"abcdefghijklmnopqrstuvwxyz"

    result = service.ingest(filename=r"..\uploads\paper.txt", content=content)

    content_hash = hashlib.sha256(content).hexdigest()
    assert result.status == "completed"
    assert result.operation == "created"
    assert result.document_id == f"doc_{content_hash[:32]}"
    assert result.content_sha256 == content_hash
    assert result.filename == "paper.txt"
    assert result.file_type == "txt"
    assert result.chunk_count == 3
    assert result.embedding_count == 3
    assert result.entity_count == 2
    assert result.relation_count == 3
    assert result.deleted_chunk_count == 0
    assert result.deleted_relation_count == 0
    assert result.deleted_entity_count == 0
    assert embedding_model.batch_sizes == [2, 1]
    assert vector_store.created_dimensions == [3]
    assert len(vector_store.upserted_records) == 3
    assert len(graph_store.records) == 3
    assert result.timings.total_ms >= 0
    assert result.timings.cleanup_ms == 0


def test_ingest_rejects_duplicate_before_expensive_work() -> None:
    vector_store = FakeVectorStore(document_exists=True)
    service, embedding_model, _, graph_store = build_service(vector_store=vector_store)

    with pytest.raises(DuplicateDocumentError) as exc_info:
        service.ingest(filename="paper.txt", content=b"duplicate")

    assert exc_info.value.document_id == build_content_document_id(hashlib.sha256(b"duplicate").hexdigest())
    assert embedding_model.batch_sizes == []
    assert graph_store.records == []


def test_ingest_rejects_unsupported_and_empty_documents() -> None:
    service, _, vector_store, _ = build_service()

    with pytest.raises(UnsupportedDocumentTypeError):
        service.ingest(filename="paper.docx", content=b"content")
    with pytest.raises(DocumentValidationError):
        service.ingest(filename="paper.txt", content=b"")

    assert vector_store.checked_document_ids == []


def test_ingest_rejects_overlong_filename() -> None:
    service, _, vector_store, _ = build_service()

    with pytest.raises(DocumentValidationError, match="512"):
        service.ingest(filename=f"{'a' * 513}.txt", content=b"content")

    assert vector_store.checked_document_ids == []


def test_graph_extraction_failure_happens_before_database_writes() -> None:
    extractor = FakeGraphExtractor(error=RuntimeError("provider failed"))
    service, _, vector_store, graph_store = build_service(graph_extractor=extractor)

    with pytest.raises(IngestionStageError) as exc_info:
        service.ingest(filename="paper.txt", content=b"content for graph extraction")

    assert exc_info.value.stage == "graph_extraction"
    assert exc_info.value.status == "failed"
    assert vector_store.upserted_records == []
    assert graph_store.records == []


def test_graph_write_failure_is_reported_as_partial() -> None:
    graph_store = FakeGraphStore(error=ConnectionError("neo4j unavailable"))
    service, _, vector_store, _ = build_service(graph_store=graph_store)

    with pytest.raises(IngestionStageError) as exc_info:
        service.ingest(filename="paper.txt", content=b"content for partial write")

    assert exc_info.value.stage == "graph_write"
    assert exc_info.value.status == "partial_failed"
    assert vector_store.upserted_records == []


def test_overwrite_cleans_existing_document_after_model_work() -> None:
    vector_store = FakeVectorStore(document_exists=True, delete_count=3)
    graph_store = FakeGraphStore(
        deletion=GraphDocumentDeletion(
            deleted_relation_count=2,
            deleted_entity_count=1,
        )
    )
    service, embedding_model, _, _ = build_service(
        vector_store=vector_store,
        graph_store=graph_store,
    )

    result = service.ingest(
        filename="paper.txt",
        content=b"replacement content",
        overwrite=True,
    )

    assert result.operation == "replaced"
    assert result.deleted_chunk_count == 3
    assert result.deleted_relation_count == 2
    assert result.deleted_entity_count == 1
    assert embedding_model.batch_sizes
    assert graph_store.deleted_document_ids == [result.document_id]
    assert vector_store.deleted_document_ids == [result.document_id]


def test_overwrite_cleanup_failure_reports_completed_delete_counts() -> None:
    vector_store = FakeVectorStore(
        document_exists=True,
        delete_error=ConnectionError("milvus unavailable"),
    )
    graph_store = FakeGraphStore(
        deletion=GraphDocumentDeletion(
            deleted_relation_count=2,
            deleted_entity_count=1,
        )
    )
    service, _, _, _ = build_service(
        vector_store=vector_store,
        graph_store=graph_store,
    )

    with pytest.raises(IngestionStageError) as exc_info:
        service.ingest(
            filename="paper.txt",
            content=b"replacement content",
            overwrite=True,
        )

    assert exc_info.value.stage == "vector_delete"
    assert exc_info.value.status == "partial_failed"
    assert exc_info.value.deleted_relation_count == 2
    assert exc_info.value.deleted_entity_count == 1
    assert graph_store.records == []


def test_document_lifecycle_deletes_graph_then_vectors() -> None:
    vector_store = FakeVectorStore(delete_count=4)
    graph_store = FakeGraphStore(
        deletion=GraphDocumentDeletion(
            deleted_relation_count=3,
            deleted_entity_count=2,
        )
    )
    service = DocumentLifecycleService(
        vector_store=vector_store,
        graph_store=graph_store,
    )

    result = service.delete("doc_123")

    assert result.status == "completed"
    assert result.deleted_chunk_count == 4
    assert result.deleted_relation_count == 3
    assert result.deleted_entity_count == 2
    assert graph_store.deleted_document_ids == ["doc_123"]
    assert vector_store.deleted_document_ids == ["doc_123"]


def test_document_lifecycle_rejects_invalid_or_missing_document() -> None:
    vector_store = FakeVectorStore()
    graph_store = FakeGraphStore()
    service = DocumentLifecycleService(
        vector_store=vector_store,
        graph_store=graph_store,
    )

    with pytest.raises(DocumentIdValidationError):
        service.delete("../unsafe")
    with pytest.raises(DocumentNotFoundError):
        service.delete("doc_missing")

    assert graph_store.deleted_document_ids == ["doc_missing"]
    assert vector_store.deleted_document_ids == ["doc_missing"]


def test_document_lifecycle_reports_partial_vector_delete() -> None:
    vector_store = FakeVectorStore(delete_error=ConnectionError("milvus unavailable"))
    graph_store = FakeGraphStore(
        deletion=GraphDocumentDeletion(
            deleted_relation_count=2,
            deleted_entity_count=1,
        )
    )
    service = DocumentLifecycleService(
        vector_store=vector_store,
        graph_store=graph_store,
    )

    with pytest.raises(DocumentDeletionStageError) as exc_info:
        service.delete("doc_123")

    assert exc_info.value.stage == "vector_delete"
    assert exc_info.value.status == "partial_failed"
    assert exc_info.value.deleted_relation_count == 2
    assert exc_info.value.deleted_entity_count == 1
