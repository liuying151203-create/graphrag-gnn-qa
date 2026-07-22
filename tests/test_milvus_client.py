import json
from pathlib import Path

import pytest

from graphrag_gnn_qa.vectorstore.milvus_client import (
    EmbeddingRecord,
    MilvusVectorStore,
    infer_embedding_dimension,
    prepare_insert_columns,
    read_embedding_records,
)


def test_read_embedding_records(tmp_path: Path) -> None:
    file_path = tmp_path / "chunk_embeddings.jsonl"
    record = {
        "chunk_id": "sample_chunk_0000",
        "document_id": "sample",
        "content": "GraphRAG connects vector search and graph traversal.",
        "source": "sample.txt",
        "file_name": "sample.txt",
        "file_type": "txt",
        "embedding": [0.1, 0.2, 0.3],
    }
    file_path.write_text(json.dumps(record), encoding="utf-8")

    records = read_embedding_records(file_path)

    assert records == [
        EmbeddingRecord(
            chunk_id="sample_chunk_0000",
            document_id="sample",
            content="GraphRAG connects vector search and graph traversal.",
            source="sample.txt",
            file_name="sample.txt",
            file_type="txt",
            embedding=[0.1, 0.2, 0.3],
        )
    ]


def test_infer_embedding_dimension() -> None:
    records = [
        EmbeddingRecord("chunk_1", "doc", "content", "source", "file.txt", "txt", [0.1, 0.2]),
        EmbeddingRecord("chunk_2", "doc", "content", "source", "file.txt", "txt", [0.3, 0.4]),
    ]

    dimension = infer_embedding_dimension(records)

    assert dimension == 2


def test_infer_embedding_dimension_rejects_empty_records() -> None:
    with pytest.raises(ValueError):
        infer_embedding_dimension([])


def test_infer_embedding_dimension_rejects_inconsistent_dimensions() -> None:
    records = [
        EmbeddingRecord("chunk_1", "doc", "content", "source", "file.txt", "txt", [0.1, 0.2]),
        EmbeddingRecord("chunk_2", "doc", "content", "source", "file.txt", "txt", [0.3]),
    ]

    with pytest.raises(ValueError):
        infer_embedding_dimension(records)


def test_prepare_insert_columns() -> None:
    records = [
        EmbeddingRecord("chunk_1", "doc", "content 1", "source", "file.txt", "txt", [0.1, 0.2]),
        EmbeddingRecord("chunk_2", "doc", "content 2", "source", "file.txt", "txt", [0.3, 0.4]),
    ]

    columns = prepare_insert_columns(records)

    assert columns == [
        ["chunk_1", "chunk_2"],
        ["doc", "doc"],
        ["content 1", "content 2"],
        ["source", "source"],
        ["file.txt", "file.txt"],
        ["txt", "txt"],
        [[0.1, 0.2], [0.3, 0.4]],
    ]


def test_milvus_store_connect_ping_and_close(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "pymilvus.connections.connect",
        lambda **kwargs: calls.append(("connect", kwargs)),
    )
    monkeypatch.setattr(
        "pymilvus.utility.list_collections",
        lambda **kwargs: calls.append(("ping", kwargs)),
    )
    monkeypatch.setattr(
        "pymilvus.connections.disconnect",
        lambda alias: calls.append(("disconnect", alias)),
    )
    store = MilvusVectorStore(host="milvus", port=19530, alias="runtime")

    store.connect()
    store.ping()
    store.close()

    assert calls == [
        ("connect", {"alias": "runtime", "host": "milvus", "port": "19530"}),
        ("ping", {"using": "runtime"}),
        ("disconnect", "runtime"),
    ]


def test_milvus_store_upserts_records(monkeypatch) -> None:
    calls = []

    class MutationResult:
        upsert_count = 2

    class FakeCollection:
        def __init__(self, **kwargs) -> None:
            calls.append(("collection", kwargs))

        def upsert(self, columns):
            calls.append(("upsert", columns))
            return MutationResult()

        def flush(self) -> None:
            calls.append(("flush", None))

    monkeypatch.setattr("pymilvus.Collection", FakeCollection)
    records = [
        EmbeddingRecord("chunk_1", "doc", "content 1", "source", "file.txt", "txt", [0.1, 0.2]),
        EmbeddingRecord("chunk_2", "doc", "content 2", "source", "file.txt", "txt", [0.3, 0.4]),
    ]
    store = MilvusVectorStore(collection_name="chunks", alias="runtime")

    count = store.upsert_records(records)

    assert count == 2
    assert calls[0] == ("collection", {"name": "chunks", "using": "runtime"})
    assert calls[1] == ("upsert", prepare_insert_columns(records))
    assert calls[2] == ("flush", None)


def test_milvus_store_checks_document_exists(monkeypatch) -> None:
    calls = []

    class FakeCollection:
        def __init__(self, **kwargs) -> None:
            calls.append(("collection", kwargs))

        def load(self) -> None:
            calls.append(("load", None))

        def query(self, **kwargs):
            calls.append(("query", kwargs))
            return [{"chunk_id": "chunk_1"}]

    monkeypatch.setattr("pymilvus.utility.has_collection", lambda *args, **kwargs: True)
    monkeypatch.setattr("pymilvus.Collection", FakeCollection)
    store = MilvusVectorStore(collection_name="chunks", alias="runtime")

    exists = store.document_exists("doc_123")

    assert exists is True
    assert calls == [
        ("collection", {"name": "chunks", "using": "runtime"}),
        ("load", None),
        (
            "query",
            {
                "expr": 'document_id == "doc_123"',
                "output_fields": ["chunk_id"],
                "limit": 1,
            },
        ),
    ]


def test_milvus_store_reports_missing_document_when_collection_is_absent(monkeypatch) -> None:
    monkeypatch.setattr("pymilvus.utility.has_collection", lambda *args, **kwargs: False)
    store = MilvusVectorStore(collection_name="chunks")

    assert store.document_exists("doc_123") is False
