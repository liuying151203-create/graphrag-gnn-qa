import json
from pathlib import Path

import pytest

from graphrag_gnn_qa.vectorstore.embedding import HashEmbeddingModel
from scripts.embed_chunks import batched, embed_chunks


def test_batched_records() -> None:
    records = [{"id": index} for index in range(5)]

    batches = batched(records, batch_size=2)

    assert batches == [[{"id": 0}, {"id": 1}], [{"id": 2}, {"id": 3}], [{"id": 4}]]


def test_batched_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError):
        batched([], batch_size=0)


def test_embed_chunks_generates_embedding_jsonl(tmp_path: Path) -> None:
    input_file = tmp_path / "chunks.jsonl"
    output_file = tmp_path / "chunk_embeddings.jsonl"
    records = [
        {
            "chunk_id": "sample_chunk_0000",
            "document_id": "sample",
            "content": "GraphRAG connects vector search and graph traversal.",
            "source": "sample.txt",
            "file_name": "sample.txt",
            "file_type": "txt",
        },
        {
            "chunk_id": "sample_chunk_0001",
            "document_id": "sample",
            "content": "GNN can encode graph topology into node representations.",
            "source": "sample.txt",
            "file_name": "sample.txt",
            "file_type": "txt",
        },
    ]
    input_file.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    embedded_count = embed_chunks(
        input_file=input_file,
        output_file=output_file,
        embedding_model=HashEmbeddingModel(dimension=8),
        batch_size=1,
    )

    output_records = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]

    assert embedded_count == 2
    assert output_records[0]["chunk_id"] == "sample_chunk_0000"
    assert output_records[0]["document_id"] == "sample"
    assert len(output_records[0]["embedding"]) == 8
    assert output_records[1]["chunk_id"] == "sample_chunk_0001"
