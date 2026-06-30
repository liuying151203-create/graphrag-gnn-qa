import re
from dataclasses import replace
from typing import Protocol

from graphrag_gnn_qa.retrieval.hybrid_result import HybridEvidence


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class EvidenceReranker(Protocol):
    def rerank(self, question: str, evidences: list[HybridEvidence], top_k: int) -> list[HybridEvidence]:
        ...


class KeywordOverlapEvidenceReranker:
    """Lightweight deterministic reranker for local development and tests."""

    def rerank(
        self,
        question: str,
        evidences: list[HybridEvidence],
        top_k: int,
    ) -> list[HybridEvidence]:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_terms = _tokenize(normalized_question)
        if not query_terms:
            return evidences[:top_k]

        scored_evidences = []
        for index, evidence in enumerate(evidences):
            matched_terms = _matched_terms(query_terms=query_terms, evidence=evidence)
            rerank_score = len(matched_terms) / len(query_terms)
            scored_evidences.append((rerank_score, evidence.fusion_score, -index, evidence, matched_terms))

        reranked_evidences = []
        for rerank_score, _, _, evidence, matched_terms in sorted(scored_evidences, reverse=True)[:top_k]:
            reranked_evidences.append(
                replace(
                    evidence,
                    metadata={
                        **evidence.metadata,
                        "rerank_score": f"{rerank_score:.6f}",
                        "rerank_matched_terms": ",".join(matched_terms),
                        "reranker_type": "keyword",
                    },
                )
            )
        return reranked_evidences


class BGEEvidenceReranker:
    """BGE/cross-encoder reranker backed by FlagEmbedding."""

    def __init__(
        self,
        model_name: str,
        use_fp16: bool = False,
        reranker_factory: type | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._reranker_factory = reranker_factory
        self._reranker = None

    def rerank(
        self,
        question: str,
        evidences: list[HybridEvidence],
        top_k: int,
    ) -> list[HybridEvidence]:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not evidences:
            return []

        pairs = [(normalized_question, _evidence_text(evidence)) for evidence in evidences]
        raw_scores = self._get_reranker().compute_score(pairs)
        scores = _coerce_scores(raw_scores)
        if len(scores) != len(evidences):
            raise ValueError("reranker returned a score count that does not match evidences")

        scored_evidences = [
            (score, evidence.fusion_score, -index, evidence)
            for index, (score, evidence) in enumerate(zip(scores, evidences))
        ]
        reranked_evidences = []
        for score, _, _, evidence in sorted(scored_evidences, reverse=True)[:top_k]:
            reranked_evidences.append(
                replace(
                    evidence,
                    metadata={
                        **evidence.metadata,
                        "rerank_score": f"{score:.6f}",
                        "reranker_type": "bge",
                        "reranker_model": self.model_name,
                    },
                )
            )
        return reranked_evidences

    def _get_reranker(self):
        if self._reranker is None:
            if self._reranker_factory is None:
                from FlagEmbedding import FlagReranker

                self._reranker_factory = FlagReranker
            self._reranker = self._reranker_factory(self.model_name, use_fp16=self.use_fp16)
        return self._reranker


class FallbackEvidenceReranker:
    def __init__(self, primary: EvidenceReranker, fallback: EvidenceReranker) -> None:
        self.primary = primary
        self.fallback = fallback

    def rerank(
        self,
        question: str,
        evidences: list[HybridEvidence],
        top_k: int,
    ) -> list[HybridEvidence]:
        try:
            return self.primary.rerank(question=question, evidences=evidences, top_k=top_k)
        except Exception as exc:
            reranked = self.fallback.rerank(question=question, evidences=evidences, top_k=top_k)
            return [
                replace(
                    evidence,
                    metadata={
                        **evidence.metadata,
                        "reranker_fallback": "keyword",
                        "reranker_fallback_reason": exc.__class__.__name__,
                    },
                )
                for evidence in reranked
            ]


def _coerce_scores(raw_scores) -> list[float]:
    if isinstance(raw_scores, int | float):
        return [float(raw_scores)]
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    return [float(score) for score in raw_scores]


def _tokenize(text: str) -> list[str]:
    terms = []
    for token in TOKEN_PATTERN.findall(text.casefold()):
        if len(token) < 2:
            continue
        if token not in terms:
            terms.append(token)
    return terms


def _matched_terms(query_terms: list[str], evidence: HybridEvidence) -> list[str]:
    evidence_text = _evidence_text(evidence)
    return [term for term in query_terms if term in evidence_text]


def _evidence_text(evidence: HybridEvidence) -> str:
    metadata_text = " ".join(str(value) for value in evidence.metadata.values())
    return f"{evidence.content} {metadata_text}".casefold()
