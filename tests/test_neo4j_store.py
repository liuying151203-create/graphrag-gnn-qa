import pytest

from graphrag_gnn_qa.graph.neo4j_store import (
    build_entity_id,
    build_entity_merge_query,
    build_export_edges_query,
    build_export_nodes_query,
    build_neighbor_search_query,
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


def test_build_neighbor_search_query() -> None:
    query = build_neighbor_search_query(max_depth=2)

    assert "MATCH path = (center)-[rel*1..2]-(neighbor)" in query
    assert "toLower(center.name) CONTAINS $query_text" in query
    assert "relationship.evidence AS evidence" in query
    assert "relationship.confidence AS confidence" in query


def test_build_neighbor_search_query_rejects_invalid_depth() -> None:
    with pytest.raises(ValueError):
        build_neighbor_search_query(max_depth=0)


def test_build_export_nodes_query() -> None:
    query = build_export_nodes_query()

    assert "MATCH (node)" in query
    assert "node.id AS node_id" in query
    assert "labels(node)[0] AS node_type" in query
    assert "ORDER BY node_id" in query


def test_build_export_edges_query() -> None:
    query = build_export_edges_query()

    assert "MATCH (source_node)-[relationship]->(target_node)" in query
    assert "source_node.id AS source_id" in query
    assert "type(relationship) AS relation_type" in query
    assert "relationship.confidence" in query


def test_validate_entity_type_rejects_unknown_type() -> None:
    with pytest.raises(ValueError):
        validate_entity_type("Algorithm")


def test_validate_relation_type_rejects_unknown_type() -> None:
    with pytest.raises(ValueError):
        validate_relation_type("MENTIONS")
