import pytest

from graphrag_gnn_qa.ingestion.text_splitter import TextSplitter


def test_split_text_into_overlapped_chunks() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"
    splitter = TextSplitter(chunk_size=10, chunk_overlap=2)

    chunks = splitter.split(text, document_id="paper")

    assert [chunk.content for chunk in chunks] == ["abcdefghij", "ijklmnopqr", "qrstuvwxyz"]
    assert [chunk.chunk_id for chunk in chunks] == [
        "paper_chunk_0000",
        "paper_chunk_0001",
        "paper_chunk_0002",
    ]


def test_split_empty_text_returns_empty_list() -> None:
    splitter = TextSplitter(chunk_size=10, chunk_overlap=2)

    chunks = splitter.split("   \n\n   ")

    assert chunks == []


def test_invalid_splitter_config() -> None:
    with pytest.raises(ValueError):
        TextSplitter(chunk_size=10, chunk_overlap=10)
