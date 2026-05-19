import pytest

from graphrag_gnn_qa.graph.neo4j_store import (
    build_entity_id,
    build_entity_merge_query,
    build_relation_merge_query,
    validate_entity_type,
    validate_relation_type,
)


def test_build_entity_id_normalizes_name() -> None:
    entity_id = build_entity_id("Method", "  Graph   RAG  ")

    assert entity_id == "Method:graph rag"


def test_build_entity_merge_query() -> None:
    query = build_entity_merge_query("Method")

    assert "MERGE (n:Method {id: $id})" in query
    assert "SET n.name = $name" in query


def test_build_relation_merge_query_keeps_evidence_properties() -> None:
    query = build_relation_merge_query("Method", "SOLVES_TASK", "Task")

    assert "MERGE (source:Method {id: $source_id})" in query
    assert "MERGE (target:Task {id: $target_id})" in query
    assert "MERGE (source)-[rel:SOLVES_TASK {chunk_id: $chunk_id}]->(target)" in query
    assert "rel.evidence = $evidence" in query
    assert "rel.confidence = $confidence" in query


def test_validate_entity_type_rejects_unknown_type() -> None:
    with pytest.raises(ValueError):
        validate_entity_type("Algorithm")


def test_validate_relation_type_rejects_unknown_type() -> None:
    with pytest.raises(ValueError):
        validate_relation_type("MENTIONS")
