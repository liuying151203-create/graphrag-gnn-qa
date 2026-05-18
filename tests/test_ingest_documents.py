import json
from pathlib import Path

from scripts.ingest_documents import build_document_id, ingest_documents


def test_build_document_id_from_nested_path(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    file_path = input_dir / "papers" / "graph rag.md"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("content", encoding="utf-8")

    document_id = build_document_id(file_path, input_dir)

    assert document_id == "papers_graph_rag"


def test_ingest_documents_generates_jsonl_chunks(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_file = tmp_path / "processed" / "chunks.jsonl"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")

    chunk_count = ingest_documents(
        input_dir=input_dir,
        output_file=output_file,
        chunk_size=10,
        chunk_overlap=2,
    )

    records = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]

    assert chunk_count == 3
    assert [record["chunk_id"] for record in records] == [
        "sample_chunk_0000",
        "sample_chunk_0001",
        "sample_chunk_0002",
    ]
    assert records[0]["content"] == "abcdefghij"
    assert records[0]["file_name"] == "sample.txt"
    assert records[0]["file_type"] == "txt"
