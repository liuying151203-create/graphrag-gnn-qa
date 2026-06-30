import pytest

from graphrag_gnn_qa.rerank import BGEEvidenceReranker, FallbackEvidenceReranker, KeywordOverlapEvidenceReranker
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


class FakeFlagReranker:
    def __init__(self, model_name: str, use_fp16: bool = False) -> None:
        self.model_name = model_name
        self.use_fp16 = use_fp16

    def compute_score(self, sentence_pairs):
        return [0.1 if "general" in pair[1] else 0.9 for pair in sentence_pairs]


class FailingReranker:
    def rerank(self, question: str, evidences: list[HybridEvidence], top_k: int) -> list[HybridEvidence]:
        raise RuntimeError("model load failed")


def test_bge_reranker_uses_model_scores_and_metadata() -> None:
    reranker = BGEEvidenceReranker(model_name="fake-reranker", reranker_factory=FakeFlagReranker)
    evidences = [
        make_evidence(evidence_id="V1", content="general GraphRAG evidence", fusion_score=0.99),
        make_evidence(evidence_id="V2", content="GraphRAG topology evidence", fusion_score=0.1),
    ]

    reranked = reranker.rerank(question="What uses topology?", evidences=evidences, top_k=2)

    assert [evidence.evidence_id for evidence in reranked] == ["V2", "V1"]
    assert reranked[0].metadata["rerank_score"] == "0.900000"
    assert reranked[0].metadata["reranker_type"] == "bge"
    assert reranked[0].metadata["reranker_model"] == "fake-reranker"


def test_fallback_reranker_uses_keyword_when_primary_fails() -> None:
    reranker = FallbackEvidenceReranker(
        primary=FailingReranker(),
        fallback=KeywordOverlapEvidenceReranker(),
    )
    evidences = [
        make_evidence(evidence_id="V1", content="general evidence", fusion_score=0.99),
        make_evidence(evidence_id="V2", content="GraphRAG topology evidence", fusion_score=0.1),
    ]

    reranked = reranker.rerank(question="GraphRAG topology", evidences=evidences, top_k=1)

    assert [evidence.evidence_id for evidence in reranked] == ["V2"]
    assert reranked[0].metadata["reranker_type"] == "keyword"
    assert reranked[0].metadata["reranker_fallback"] == "keyword"
    assert reranked[0].metadata["reranker_fallback_reason"] == "RuntimeError"
