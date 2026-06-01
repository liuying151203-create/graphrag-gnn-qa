import pytest

from graphrag_gnn_qa.rerank import KeywordOverlapEvidenceReranker
from graphrag_gnn_qa.retrieval.hybrid_result import EvidenceType, HybridEvidence


def make_evidence(evidence_id: str, content: str, fusion_score: float) -> HybridEvidence:
    return HybridEvidence(
        evidence_id=evidence_id,
        evidence_type=EvidenceType.VECTOR_CHUNK,
        rank=1,
        score=fusion_score,
        fusion_score=fusion_score,
        document_id="sample",
        chunk_id=evidence_id,
        source="sample.txt",
        content=content,
        metadata={},
    )


def test_keyword_overlap_reranker_prioritizes_question_terms() -> None:
    reranker = KeywordOverlapEvidenceReranker()
    evidences = [
        make_evidence(
            evidence_id="V1",
            content="GraphRAG retrieves general evidence.",
            fusion_score=0.99,
        ),
        make_evidence(
            evidence_id="V2",
            content="A Graph Attention Network uses graph topology for node embeddings.",
            fusion_score=0.2,
        ),
    ]

    reranked = reranker.rerank(
        question="How does Graph Attention Network use graph topology?",
        evidences=evidences,
        top_k=2,
    )

    assert [evidence.evidence_id for evidence in reranked] == ["V2", "V1"]
    assert reranked[0].metadata["rerank_score"] == "0.714286"
    assert "topology" in reranked[0].metadata["rerank_matched_terms"]


def test_keyword_overlap_reranker_uses_fusion_score_as_tiebreaker() -> None:
    reranker = KeywordOverlapEvidenceReranker()
    evidences = [
        make_evidence(evidence_id="V1", content="GraphRAG evidence.", fusion_score=0.3),
        make_evidence(evidence_id="V2", content="GraphRAG evidence.", fusion_score=0.8),
    ]

    reranked = reranker.rerank(question="GraphRAG", evidences=evidences, top_k=2)

    assert [evidence.evidence_id for evidence in reranked] == ["V2", "V1"]


def test_keyword_overlap_reranker_respects_top_k() -> None:
    reranker = KeywordOverlapEvidenceReranker()

    reranked = reranker.rerank(
        question="GraphRAG",
        evidences=[
            make_evidence(evidence_id="V1", content="GraphRAG one.", fusion_score=0.1),
            make_evidence(evidence_id="V2", content="GraphRAG two.", fusion_score=0.2),
        ],
        top_k=1,
    )

    assert len(reranked) == 1
    assert reranked[0].evidence_id == "V2"


def test_keyword_overlap_reranker_rejects_invalid_inputs() -> None:
    reranker = KeywordOverlapEvidenceReranker()

    with pytest.raises(ValueError, match="question must not be empty"):
        reranker.rerank(question=" ", evidences=[], top_k=1)

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        reranker.rerank(question="GraphRAG", evidences=[], top_k=0)
