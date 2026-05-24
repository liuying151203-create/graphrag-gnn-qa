import pytest

from graphrag_gnn_qa.retrieval.graph_retriever import RetrievedGraphRelation
from graphrag_gnn_qa.retrieval.hybrid_result import (
    EvidenceType,
    apply_fusion_scores,
    build_hybrid_evidences,
    build_hybrid_retrieval_result,
    chunk_to_hybrid_evidence,
    graph_relation_to_hybrid_evidence,
    normalize_scores,
    rank_hybrid_evidences,
)
from graphrag_gnn_qa.retrieval.vector_retriever import RetrievedChunk


def sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        score=0.91,
        chunk_id="sample_chunk_0000",
        document_id="sample",
        content="GraphRAG connects vector search and graph traversal.",
        source="sample.txt",
        file_name="sample.txt",
        file_type="txt",
    )


def sample_graph_relation() -> RetrievedGraphRelation:
    return RetrievedGraphRelation(
        center_id="Method:graphrag",
        center_name="GraphRAG",
        center_type="Method",
        source_id="Method:graphrag",
        source_name="GraphRAG",
        source_type="Method",
        relation_type="SOLVES_TASK",
        target_id="Task:question answering",
        target_name="question answering",
        target_type="Task",
        chunk_id="sample_chunk_0000",
        document_id="sample",
        source="sample.txt",
        evidence="GraphRAG improves question answering.",
        confidence=0.9,
    )


def test_chunk_to_hybrid_evidence() -> None:
    evidence = chunk_to_hybrid_evidence(chunk=sample_chunk(), rank=1)

    assert evidence.evidence_id == "V1"
    assert evidence.evidence_type == EvidenceType.VECTOR_CHUNK
    assert evidence.rank == 1
    assert evidence.score == 0.91
    assert evidence.fusion_score == 0.91
    assert evidence.document_id == "sample"
    assert evidence.chunk_id == "sample_chunk_0000"
    assert evidence.source == "sample.txt"
    assert evidence.content == "GraphRAG connects vector search and graph traversal."
    assert evidence.metadata == {"file_name": "sample.txt", "file_type": "txt"}


def test_graph_relation_to_hybrid_evidence() -> None:
    evidence = graph_relation_to_hybrid_evidence(relation=sample_graph_relation(), rank=1)

    assert evidence.evidence_id == "G1"
    assert evidence.evidence_type == EvidenceType.GRAPH_RELATION
    assert evidence.rank == 1
    assert evidence.score == 0.9
    assert evidence.fusion_score == 0.9
    assert evidence.document_id == "sample"
    assert evidence.chunk_id == "sample_chunk_0000"
    assert evidence.source == "sample.txt"
    assert evidence.content == "GraphRAG improves question answering."
    assert evidence.metadata["center_name"] == "GraphRAG"
    assert evidence.metadata["relation_type"] == "SOLVES_TASK"
    assert evidence.metadata["target_name"] == "question answering"


def test_build_hybrid_evidences_combines_vector_and_graph_evidence() -> None:
    evidences = build_hybrid_evidences(chunks=[sample_chunk()], graph_relations=[sample_graph_relation()])

    assert [evidence.evidence_id for evidence in evidences] == ["V1", "G1"]
    assert [evidence.evidence_type for evidence in evidences] == [EvidenceType.VECTOR_CHUNK, EvidenceType.GRAPH_RELATION]
    assert [evidence.fusion_score for evidence in evidences] == [0.937, 0.93]


def test_normalize_scores_scales_scores_to_zero_one_range() -> None:
    assert normalize_scores([0.2, 0.6, 1.0]) == [0.0, 0.49999999999999994, 1.0]


def test_normalize_scores_clamps_single_or_equal_score_groups() -> None:
    assert normalize_scores([]) == []
    assert normalize_scores([1.2]) == [1.0]
    assert normalize_scores([-0.2, 1.2]) == [0.0, 1.0]


def test_apply_fusion_scores_combines_normalized_score_and_rank_score() -> None:
    evidences = [
        chunk_to_hybrid_evidence(chunk=sample_chunk(), rank=2),
        graph_relation_to_hybrid_evidence(relation=sample_graph_relation(), rank=1),
    ]

    fused_evidences = apply_fusion_scores(evidences=evidences)

    assert [evidence.fusion_score for evidence in fused_evidences] == [0.787, 0.93]


def test_rank_hybrid_evidences_sorts_by_fusion_score_descending() -> None:
    evidences = apply_fusion_scores(
        [
            chunk_to_hybrid_evidence(chunk=sample_chunk(), rank=2),
            graph_relation_to_hybrid_evidence(relation=sample_graph_relation(), rank=1),
        ]
    )

    ranked_evidences = rank_hybrid_evidences(evidences)

    assert [evidence.evidence_id for evidence in ranked_evidences] == ["G1", "V2"]


def test_build_hybrid_retrieval_result() -> None:
    result = build_hybrid_retrieval_result(
        query="What is GraphRAG?",
        chunks=[sample_chunk()],
        graph_relations=[sample_graph_relation()],
    )

    assert result.query == "What is GraphRAG?"
    assert len(result.evidences) == 2


def test_build_hybrid_retrieval_result_rejects_empty_query() -> None:
    with pytest.raises(ValueError):
        build_hybrid_retrieval_result(query="   ", chunks=[], graph_relations=[])


def test_hybrid_evidence_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError):
        chunk_to_hybrid_evidence(chunk=sample_chunk(), rank=0)
    with pytest.raises(ValueError):
        graph_relation_to_hybrid_evidence(relation=sample_graph_relation(), rank=0)


def test_apply_fusion_scores_rejects_invalid_weights() -> None:
    evidences = [chunk_to_hybrid_evidence(chunk=sample_chunk(), rank=1)]

    with pytest.raises(ValueError):
        apply_fusion_scores(evidences=evidences, score_weight=-1)
    with pytest.raises(ValueError):
        apply_fusion_scores(evidences=evidences, rank_weight=-1)
    with pytest.raises(ValueError):
        apply_fusion_scores(evidences=evidences, score_weight=0, rank_weight=0)
