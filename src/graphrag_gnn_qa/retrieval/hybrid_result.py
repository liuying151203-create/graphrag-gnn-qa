from dataclasses import dataclass, replace
from enum import Enum

from graphrag_gnn_qa.retrieval.graph_retriever import RetrievedGraphRelation
from graphrag_gnn_qa.retrieval.vector_retriever import RetrievedChunk

DEFAULT_FUSION_SCORE_WEIGHT = 0.7
DEFAULT_FUSION_RANK_WEIGHT = 0.3


class EvidenceType(str, Enum):
    VECTOR_CHUNK = "vector_chunk"
    GRAPH_RELATION = "graph_relation"


@dataclass(frozen=True)
class HybridEvidence:
    evidence_id: str
    evidence_type: EvidenceType
    rank: int
    score: float
    fusion_score: float
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
        fusion_score=chunk.score,
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
        fusion_score=relation.confidence,
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


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    if len(scores) == 1:
        return [_clamp_score(scores[0])]

    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [_clamp_score(score) for score in scores]

    return [(score - min_score) / (max_score - min_score) for score in scores]


def apply_fusion_scores(
    evidences: list[HybridEvidence],
    score_weight: float = DEFAULT_FUSION_SCORE_WEIGHT,
    rank_weight: float = DEFAULT_FUSION_RANK_WEIGHT,
) -> list[HybridEvidence]:
    if score_weight < 0:
        raise ValueError("score_weight must not be negative")
    if rank_weight < 0:
        raise ValueError("rank_weight must not be negative")
    if score_weight + rank_weight <= 0:
        raise ValueError("score_weight and rank_weight must not both be zero")

    normalized_scores_by_index: dict[int, float] = {}
    for evidence_type in EvidenceType:
        indexed_evidences = [
            (index, evidence) for index, evidence in enumerate(evidences) if evidence.evidence_type == evidence_type
        ]
        normalized_scores = normalize_scores([evidence.score for _, evidence in indexed_evidences])
        for (index, _), normalized_score in zip(indexed_evidences, normalized_scores):
            normalized_scores_by_index[index] = normalized_score

    total_weight = score_weight + rank_weight
    normalized_score_weight = score_weight / total_weight
    normalized_rank_weight = rank_weight / total_weight

    fused_evidences = []
    for index, evidence in enumerate(evidences):
        if evidence.rank <= 0:
            raise ValueError("rank must be greater than 0")
        fusion_score = (
            normalized_score_weight * normalized_scores_by_index[index]
            + normalized_rank_weight * (1.0 / evidence.rank)
        )
        fused_evidences.append(replace(evidence, fusion_score=round(fusion_score, 6)))
    return fused_evidences


def rank_hybrid_evidences(evidences: list[HybridEvidence]) -> list[HybridEvidence]:
    return sorted(evidences, key=lambda evidence: evidence.fusion_score, reverse=True)


def build_hybrid_evidences(
    chunks: list[RetrievedChunk],
    graph_relations: list[RetrievedGraphRelation],
) -> list[HybridEvidence]:
    vector_evidences = [chunk_to_hybrid_evidence(chunk=chunk, rank=index) for index, chunk in enumerate(chunks, start=1)]
    graph_evidences = [
        graph_relation_to_hybrid_evidence(relation=relation, rank=index)
        for index, relation in enumerate(graph_relations, start=1)
    ]
    return rank_hybrid_evidences(apply_fusion_scores(vector_evidences + graph_evidences))


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))


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
