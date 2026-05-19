import pytest

from graphrag_gnn_qa.retrieval.query_entities import extract_query_entities


def test_extract_query_entities_returns_mixed_case_term() -> None:
    assert "GraphRAG" in extract_query_entities("What is GraphRAG?")


def test_extract_query_entities_returns_keyword_phrase() -> None:
    entities = extract_query_entities("How does GraphRAG solve information fragmentation?")

    assert "GraphRAG" in entities
    assert "information fragmentation" in entities


def test_extract_query_entities_returns_quoted_phrase() -> None:
    assert extract_query_entities('What is "information fragmentation"?')[0] == "information fragmentation"


def test_extract_query_entities_rejects_empty_question() -> None:
    with pytest.raises(ValueError):
        extract_query_entities("   ")


def test_extract_query_entities_rejects_invalid_max_entities() -> None:
    with pytest.raises(ValueError):
        extract_query_entities("GraphRAG", max_entities=0)
