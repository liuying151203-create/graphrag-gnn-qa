from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    name: str
    node_type: str
    description: str = ""


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    relation_type: str
    chunk_id: str = ""
    document_id: str = ""
    source: str = ""
    evidence: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class GraphDataset:
    nodes: list[GraphNode]
    edges: list[GraphEdge]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
            "summary": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
            },
        }


def graph_dataset_from_records(
    node_records: list[dict[str, Any]],
    edge_records: list[dict[str, Any]],
) -> GraphDataset:
    nodes = [
        GraphNode(
            node_id=str(record.get("node_id", "")),
            name=str(record.get("name", "")),
            node_type=str(record.get("node_type", "")),
            description=str(record.get("description") or ""),
        )
        for record in node_records
    ]
    edges = [
        GraphEdge(
            source_id=str(record.get("source_id", "")),
            target_id=str(record.get("target_id", "")),
            relation_type=str(record.get("relation_type", "")),
            chunk_id=str(record.get("chunk_id") or ""),
            document_id=str(record.get("document_id") or ""),
            source=str(record.get("source") or ""),
            evidence=str(record.get("evidence") or ""),
            confidence=float(record.get("confidence") or 0.0),
        )
        for record in edge_records
    ]
    return GraphDataset(nodes=nodes, edges=edges)
