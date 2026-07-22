from pathlib import Path

import pytest

from graphrag_gnn_qa.ingestion.document_loader import DocumentLoader


def test_load_text_document(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("GraphRAG connects vector search with graph traversal.", encoding="utf-8")

    document = DocumentLoader().load(file_path)

    assert document.content == "GraphRAG connects vector search with graph traversal."
    assert document.file_name == "sample.txt"
    assert document.file_type == "txt"


def test_load_unsupported_document_type(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.docx"
    file_path.write_text("unsupported", encoding="utf-8")

    with pytest.raises(ValueError):
        DocumentLoader().load(file_path)


def test_load_text_document_from_bytes() -> None:
    document = DocumentLoader().load_bytes(
        content=b"\xef\xbb\xbfGraphRAG content",
        file_name="uploaded.md",
    )

    assert document.content == "GraphRAG content"
    assert document.source == "uploaded.md"
    assert document.file_type == "md"


def test_load_bytes_rejects_non_utf8_text() -> None:
    with pytest.raises(ValueError, match="UTF-8"):
        DocumentLoader().load_bytes(content=b"\xff\xfe", file_name="uploaded.txt")


def test_load_bytes_rejects_invalid_pdf() -> None:
    with pytest.raises(ValueError, match="Failed to parse PDF"):
        DocumentLoader().load_bytes(content=b"not a pdf", file_name="uploaded.pdf")
