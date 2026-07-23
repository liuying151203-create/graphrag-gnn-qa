import pytest

from graphrag_gnn_qa.graph.neo4j_store import (
    GraphDocumentDeletion,
    Neo4jGraphStore,
    _delete_document_transaction,
    build_cleanup_document_entities_query,
    build_delete_document_relations_query,
    build_entity_id,
    build_entity_merge_query,
    build_export_edges_query,
    build_export_nodes_query,
    build_neighbor_search_query,
    build_relation_merge_query,
    validate_entity_type,
    validate_relation_type,
)


class FakeDriver:
    def __init__(self) -> None:
        self.verify_count = 0
        self.close_count = 0

    def verify_connectivity(self) -> None:
        self.verify_count += 1

    def close(self) -> None:
        self.close_count += 1


def test_neo4j_store_ping_and_close(monkeypatch) -> None:
    driver = FakeDriver()
    monkeypatch.setattr(
        "graphrag_gnn_qa.graph.neo4j_store.GraphDatabase.driver",
        lambda *args, **kwargs: driver,
    )
    store = Neo4jGraphStore(uri="bolt://neo4j", username="neo4j", password="password")

    store.ping()
    store.close()

    assert driver.verify_count == 1
    assert driver.close_count == 1


def test_build_entity_id_normalizes_name() -> None:
    entity_id = build_entity_id("Method", "  Graph   RAG  ")

    assert entity_id == "Method:graph rag"


def test_build_entity_merge_query() -> None:
    query = build_entity_merge_query("Method")

    assert "MERGE (n:Method {id: $id})" in query
    assert "SET n.name = $name" in query
    assert "$document_id IN coalesce(n.document_ids, [])" in query


def test_build_relation_merge_query_keeps_evidence_properties() -> None:
    query = build_relation_merge_query("Method", "SOLVES_TASK", "Task")

    assert "MERGE (source:Method {id: $source_id})" in query
    assert "MERGE (target:Task {id: $target_id})" in query
    assert (
        "MERGE (source)-[rel:SOLVES_TASK "
        "{document_id: $document_id, chunk_id: $chunk_id}]->(target)"
    ) in query
    assert "rel.evidence = $evidence" in query
    assert "rel.confidence = $confidence" in query
    assert "source.document_ids" in query
    assert "target.document_ids" in query


def test_build_document_deletion_queries_preserve_shared_entities() -> None:
    relation_query = build_delete_document_relations_query()
    entity_query = build_cleanup_document_entities_query()

    assert "relationship.document_id = $document_id" in relation_query
    assert "DELETE relationship" in relation_query
    assert "$document_id IN coalesce(node.document_ids, [])" in entity_query
    assert "item <> $document_id" in entity_query
    assert "size(node.document_ids) = 0" in entity_query
    assert "NOT EXISTS { MATCH (node)--() }" in entity_query


def test_delete_document_transaction_returns_counts() -> None:
    class FakeResult:
        def __init__(self, record: dict) -> None:
            self.record = record

        def single(self) -> dict:
            return self.record

    class FakeTransaction:
        def __init__(self) -> None:
            self.calls = []

        def run(self, query: str, **parameters):
            self.calls.append((query, parameters))
            if "deleted_relation_count" in query:
                return FakeResult({"deleted_relation_count": 4})
            return FakeResult({"deleted_entity_count": 2})

    transaction = FakeTransaction()

    result = _delete_document_transaction(transaction, "doc_123")

    assert result == GraphDocumentDeletion(
        deleted_relation_count=4,
        deleted_entity_count=2,
    )
    assert [parameters for _, parameters in transaction.calls] == [
        {"document_id": "doc_123"},
        {"document_id": "doc_123"},
    ]


def test_neo4j_store_delete_uses_write_transaction(monkeypatch) -> None:
    expected = GraphDocumentDeletion(deleted_relation_count=3, deleted_entity_count=1)
    calls = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def execute_write(self, callback, document_id: str):
            calls.append((callback, document_id))
            return expected

    class DeleteDriver(FakeDriver):
        def session(self, **kwargs):
            calls.append(("session", kwargs))
            return FakeSession()

    driver = DeleteDriver()
    monkeypatch.setattr(
        "graphrag_gnn_qa.graph.neo4j_store.GraphDatabase.driver",
        lambda *args, **kwargs: driver,
    )
    store = Neo4jGraphStore(uri="bolt://neo4j", username="neo4j", password="password")

    result = store.delete_document("doc_123")

    assert result == expected
    assert calls[0] == ("session", {"database": "neo4j"})
    assert calls[1] == (_delete_document_transaction, "doc_123")


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
