import json
from dataclasses import dataclass
from typing import Any

from graphrag_gnn_qa.llm.client import LLMClient


ALLOWED_ENTITY_TYPES = {"Paper", "Author", "Institution", "Method", "Task", "Dataset", "Metric", "Concept"}
ALLOWED_RELATION_TYPES = {
    "AUTHORED_BY",
    "AFFILIATED_WITH",
    "PROPOSED_METHOD",
    "USES_METHOD",
    "SOLVES_TASK",
    "USES_DATASET",
    "EVALUATED_BY",
    "RELATED_TO",
}
DEFAULT_ENTITY_TYPE = "Concept"
DEFAULT_RELATION_TYPE = "RELATED_TO"


@dataclass(frozen=True)
class GraphEntity:
    name: str
    type: str
    description: str = ""


@dataclass(frozen=True)
class GraphRelation:
    source_entity: str
    source_type: str
    relation_type: str
    target_entity: str
    target_type: str
    evidence: str
    confidence: float


@dataclass(frozen=True)
class GraphExtractionResult:
    chunk_id: str
    document_id: str
    source: str
    entities: list[GraphEntity]
    relations: list[GraphRelation]


class GraphExtractor:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def extract_from_chunk(self, chunk: dict[str, Any]) -> GraphExtractionResult:
        prompt = build_extraction_prompt(chunk["content"])
        response = self.llm_client.generate(prompt)
        entities, relations = parse_extraction_response(response)
        return GraphExtractionResult(
            chunk_id=chunk["chunk_id"],
            document_id=chunk["document_id"],
            source=chunk["source"],
            entities=entities,
            relations=relations,
        )


def build_extraction_prompt(text: str) -> str:
    return (
        "Extract a knowledge graph from the following scientific or technical text.\n"
        "Return only valid JSON without markdown fences.\n"
        "Entity types must be one of: Paper, Author, Institution, Method, Task, Dataset, Metric, Concept.\n"
        "Relation types must be one of: AUTHORED_BY, AFFILIATED_WITH, PROPOSED_METHOD, USES_METHOD, SOLVES_TASK, USES_DATASET, EVALUATED_BY, RELATED_TO.\n"
        "Use this schema:\n"
        "{\"entities\":[{\"name\":\"...\",\"type\":\"Method\",\"description\":\"...\"}],"
        "\"relations\":[{\"source_entity\":\"...\",\"source_type\":\"Method\",\"relation_type\":\"SOLVES_TASK\",\"target_entity\":\"...\",\"target_type\":\"Task\",\"evidence\":\"...\",\"confidence\":0.8}]}\n\n"
        f"Text:\n{text}"
    )


def parse_extraction_response(response: str) -> tuple[list[GraphEntity], list[GraphRelation]]:
    data = json.loads(strip_json_markdown(response))
    entities = [parse_entity(entity) for entity in data.get("entities", [])]
    relations = [parse_relation(relation) for relation in data.get("relations", [])]
    return entities, relations


def strip_json_markdown(response: str) -> str:
    content = response.strip()
    if content.startswith("```json"):
        content = content.removeprefix("```json").strip()
    elif content.startswith("```"):
        content = content.removeprefix("```").strip()
    if content.endswith("```"):
        content = content.removesuffix("```").strip()
    return content


def parse_entity(entity: dict[str, Any]) -> GraphEntity:
    entity_type = normalize_entity_type(entity.get("type"))
    return GraphEntity(
        name=entity["name"].strip(),
        type=entity_type,
        description=entity.get("description", "").strip(),
    )


def parse_relation(relation: dict[str, Any]) -> GraphRelation:
    source_type = normalize_entity_type(relation.get("source_type"))
    target_type = normalize_entity_type(relation.get("target_type"))
    relation_type = normalize_relation_type(relation.get("relation_type"))
    return GraphRelation(
        source_entity=relation["source_entity"].strip(),
        source_type=source_type,
        relation_type=relation_type,
        target_entity=relation["target_entity"].strip(),
        target_type=target_type,
        evidence=relation.get("evidence", "").strip(),
        confidence=float(relation.get("confidence", 0.0)),
    )


def normalize_entity_type(entity_type: Any) -> str:
    entity_type_text = str(entity_type or "").strip()
    if entity_type_text in ALLOWED_ENTITY_TYPES:
        return entity_type_text
    return DEFAULT_ENTITY_TYPE


def normalize_relation_type(relation_type: Any) -> str:
    relation_type_text = str(relation_type or "").strip()
    if relation_type_text in ALLOWED_RELATION_TYPES:
        return relation_type_text
    return DEFAULT_RELATION_TYPE


def graph_result_to_dict(result: GraphExtractionResult) -> dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "document_id": result.document_id,
        "source": result.source,
        "entities": [entity.__dict__ for entity in result.entities],
        "relations": [relation.__dict__ for relation in result.relations],
    }
