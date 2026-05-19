from dataclasses import dataclass
from typing import Any, Protocol


class GraphSearchStore(Protocol):
    def search_neighbors(self, query: str, top_k: int = 5, max_depth: int = 1) -> list[dict[str, Any]]:
        ...


@dataclass(frozen=True)
class RetrievedGraphRelation:
    center_id: str
    center_name: str
    center_type: str
    source_id: str
    source_name: str
    source_type: str
    relation_type: str
    target_id: str
    target_name: str
    target_type: str
    chunk_id: str
    document_id: str
    source: str
    evidence: str
    confidence: float


class GraphRetriever:
    def __init__(self, graph_store: GraphSearchStore) -> None:
        self.graph_store = graph_store

    def retrieve(self, query: str, top_k: int = 5, max_depth: int = 1) -> list[RetrievedGraphRelation]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than 0")

        results = self.graph_store.search_neighbors(query=normalized_query, top_k=top_k, max_depth=max_depth)
        return [
            RetrievedGraphRelation(
                center_id=result["center_id"],
                center_name=result["center_name"],
                center_type=result["center_type"],
                source_id=result["source_id"],
                source_name=result["source_name"],
                source_type=result["source_type"],
                relation_type=result["relation_type"],
                target_id=result["target_id"],
                target_name=result["target_name"],
                target_type=result["target_type"],
                chunk_id=result.get("chunk_id", ""),
                document_id=result.get("document_id", ""),
                source=result.get("source", ""),
                evidence=result.get("evidence", ""),
                confidence=float(result.get("confidence", 0.0)),
            )
            for result in results
        ]
