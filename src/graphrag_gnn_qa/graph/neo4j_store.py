import re
from typing import Any

from neo4j import GraphDatabase

from graphrag_gnn_qa.graph.extractor import ALLOWED_ENTITY_TYPES, ALLOWED_RELATION_TYPES


class Neo4jGraphStore:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j") -> None:
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def create_constraints(self) -> None:
        with self.driver.session(database=self.database) as session:
            for label in sorted(ALLOWED_ENTITY_TYPES):
                session.run(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")

    def upsert_graph_record(self, record: dict[str, Any]) -> None:
        with self.driver.session(database=self.database) as session:
            for entity in record.get("entities", []):
                session.run(
                    build_entity_merge_query(entity["type"]),
                    id=build_entity_id(entity["type"], entity["name"]),
                    name=entity["name"],
                    description=entity.get("description", ""),
                )

            for relation in record.get("relations", []):
                session.run(
                    build_relation_merge_query(
                        source_type=relation["source_type"],
                        relation_type=relation["relation_type"],
                        target_type=relation["target_type"],
                    ),
                    source_id=build_entity_id(relation["source_type"], relation["source_entity"]),
                    source_name=relation["source_entity"],
                    target_id=build_entity_id(relation["target_type"], relation["target_entity"]),
                    target_name=relation["target_entity"],
                    chunk_id=record["chunk_id"],
                    document_id=record["document_id"],
                    source=record["source"],
                    evidence=relation.get("evidence", ""),
                    confidence=float(relation.get("confidence", 0.0)),
                )

    def upsert_graph_records(self, records: list[dict[str, Any]], create_constraints: bool = True) -> int:
        if create_constraints:
            self.create_constraints()
        for record in records:
            self.upsert_graph_record(record)
        return len(records)

    def search_neighbors(self, query: str, top_k: int = 5, max_depth: int = 1) -> list[dict[str, Any]]:
        query_text = query.strip().lower()
        if not query_text:
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than 0")

        with self.driver.session(database=self.database) as session:
            result = session.run(build_neighbor_search_query(max_depth), query_text=query_text, top_k=top_k)
            return [dict(record) for record in result]


def build_entity_id(entity_type: str, name: str) -> str:
    validate_entity_type(entity_type)
    normalized_name = re.sub(r"\s+", " ", name.strip()).lower()
    return f"{entity_type}:{normalized_name}"


def build_entity_merge_query(entity_type: str) -> str:
    validate_entity_type(entity_type)
    return f"MERGE (n:{entity_type} {{id: $id}}) SET n.name = $name, n.description = $description"


def build_relation_merge_query(source_type: str, relation_type: str, target_type: str) -> str:
    validate_entity_type(source_type)
    validate_entity_type(target_type)
    validate_relation_type(relation_type)
    return (
        f"MERGE (source:{source_type} {{id: $source_id}}) "
        "SET source.name = $source_name "
        f"MERGE (target:{target_type} {{id: $target_id}}) "
        "SET target.name = $target_name "
        f"MERGE (source)-[rel:{relation_type} {{chunk_id: $chunk_id}}]->(target) "
        "SET rel.document_id = $document_id, "
        "rel.source = $source, "
        "rel.evidence = $evidence, "
        "rel.confidence = $confidence"
    )


def build_neighbor_search_query(max_depth: int) -> str:
    if max_depth <= 0:
        raise ValueError("max_depth must be greater than 0")
    return (
        f"MATCH path = (center)-[rel*1..{max_depth}]-(neighbor) "
        "WHERE toLower(center.name) CONTAINS $query_text OR toLower(center.id) CONTAINS $query_text "
        "WITH center, relationships(path) AS rels "
        "LIMIT $top_k "
        "UNWIND rels AS relationship "
        "WITH center, relationship, startNode(relationship) AS source_node, endNode(relationship) AS target_node "
        "RETURN center.id AS center_id, "
        "center.name AS center_name, "
        "labels(center)[0] AS center_type, "
        "source_node.id AS source_id, "
        "source_node.name AS source_name, "
        "labels(source_node)[0] AS source_type, "
        "type(relationship) AS relation_type, "
        "target_node.id AS target_id, "
        "target_node.name AS target_name, "
        "labels(target_node)[0] AS target_type, "
        "relationship.chunk_id AS chunk_id, "
        "relationship.document_id AS document_id, "
        "relationship.source AS source, "
        "relationship.evidence AS evidence, "
        "relationship.confidence AS confidence"
    )


def validate_entity_type(entity_type: str) -> None:
    if entity_type not in ALLOWED_ENTITY_TYPES:
        raise ValueError(f"Unsupported entity type: {entity_type}")


def validate_relation_type(relation_type: str) -> None:
    if relation_type not in ALLOWED_RELATION_TYPES:
        raise ValueError(f"Unsupported relation type: {relation_type}")
