from dataclasses import dataclass
from enum import Enum

from graphrag_gnn_qa.retrieval.graph_retriever import RetrievedGraphRelation
from graphrag_gnn_qa.retrieval.vector_retriever import RetrievedChunk


class EvidenceType(str, Enum):
    VECTOR_CHUNK = "vector_chunk"
    GRAPH_RELATION = "graph_relation"


@dataclass(frozen=True)
class HybridEvidence:
    evidence_id: str
    evidence_type: EvidenceType
    rank: int
    score: float
    document_id: str
    chunk_id: str
    source: str
    content: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class HybridRetrievalResult:
    query: str
    evidences: list[HybridEvidence]


def chunk_to_hybrid_evidence(chunk: RetrievedChunk, rank: int) -> HybridEvidence:
    if rank <= 0:
        raise ValueError("rank must be greater than 0")
    return HybridEvidence(
        evidence_id=f"V{rank}",
        evidence_type=EvidenceType.VECTOR_CHUNK,
        rank=rank,
        score=chunk.score,
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        source=chunk.source,
        content=chunk.content,
        metadata={
            "file_name": chunk.file_name,
            "file_type": chunk.file_type,
        },
    )


def graph_relation_to_hybrid_evidence(relation: RetrievedGraphRelation, rank: int) -> HybridEvidence:
    if rank <= 0:
        raise ValueError("rank must be greater than 0")
    return HybridEvidence(
        evidence_id=f"G{rank}",
        evidence_type=EvidenceType.GRAPH_RELATION,
        rank=rank,
        score=relation.confidence,
        document_id=relation.document_id,
        chunk_id=relation.chunk_id,
        source=relation.source,
        content=relation.evidence,
        metadata={
            "center_id": relation.center_id,
            "center_name": relation.center_name,
            "center_type": relation.center_type,
            "source_id": relation.source_id,
            "source_name": relation.source_name,
            "source_type": relation.source_type,
            "relation_type": relation.relation_type,
            "target_id": relation.target_id,
            "target_name": relation.target_name,
            "target_type": relation.target_type,
        },
    )


def build_hybrid_evidences(
    chunks: list[RetrievedChunk],
    graph_relations: list[RetrievedGraphRelation],
) -> list[HybridEvidence]:
    vector_evidences = [chunk_to_hybrid_evidence(chunk=chunk, rank=index) for index, chunk in enumerate(chunks, start=1)]
    graph_evidences = [
        graph_relation_to_hybrid_evidence(relation=relation, rank=index)
        for index, relation in enumerate(graph_relations, start=1)
    ]
    return vector_evidences + graph_evidences


def build_hybrid_retrieval_result(
    query: str,
    chunks: list[RetrievedChunk],
    graph_relations: list[RetrievedGraphRelation],
) -> HybridRetrievalResult:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")
    return HybridRetrievalResult(
        query=normalized_query,
        evidences=build_hybrid_evidences(chunks=chunks, graph_relations=graph_relations),
    )
