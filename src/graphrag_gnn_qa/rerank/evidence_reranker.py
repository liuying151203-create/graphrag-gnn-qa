import re
from dataclasses import replace

from graphrag_gnn_qa.retrieval.hybrid_result import HybridEvidence


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


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
                    },
                )
            )
        return reranked_evidences


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
