from dataclasses import dataclass
from typing import Protocol

from graphrag_gnn_qa.llm.client import LLMClient
from graphrag_gnn_qa.rag.context_builder import build_hybrid_rag_prompt, build_rag_prompt
from graphrag_gnn_qa.retrieval.graph_retriever import RetrievedGraphRelation
from graphrag_gnn_qa.retrieval.hybrid_result import build_hybrid_retrieval_result
from graphrag_gnn_qa.retrieval.query_entities import extract_query_entities
from graphrag_gnn_qa.retrieval.vector_retriever import RetrievedChunk


class ChunkRetriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        ...


class GraphRelationRetriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5, max_depth: int = 1) -> list[RetrievedGraphRelation]:
        ...


@dataclass(frozen=True)
class SourceEvidence:
    chunk_id: str
    document_id: str
    source: str
    file_name: str
    score: float
    content: str


@dataclass(frozen=True)
class GraphEvidence:
    center_name: str
    center_type: str
    source_name: str
    source_type: str
    relation_type: str
    target_name: str
    target_type: str
    chunk_id: str
    document_id: str
    source: str
    evidence: str
    confidence: float


@dataclass(frozen=True)
class QAResult:
    question: str
    answer: str
    sources: list[SourceEvidence]
    graph_sources: list[GraphEvidence]


class RAGQAService:
    def __init__(
        self,
        retriever: ChunkRetriever,
        llm_client: LLMClient,
        graph_retriever: GraphRelationRetriever | None = None,
        graph_top_k: int = 5,
        graph_max_depth: int = 1,
    ) -> None:
        if graph_top_k <= 0:
            raise ValueError("graph_top_k must be greater than 0")
        if graph_max_depth <= 0:
            raise ValueError("graph_max_depth must be greater than 0")
        self.retriever = retriever
        self.llm_client = llm_client
        self.graph_retriever = graph_retriever
        self.graph_top_k = graph_top_k
        self.graph_max_depth = graph_max_depth

    def answer(self, question: str, top_k: int = 5) -> QAResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        chunks = self.retriever.retrieve(query=normalized_question, top_k=top_k)
        graph_relations = []
        if self.graph_retriever is not None:
            graph_relations = retrieve_graph_relations_for_question(
                graph_retriever=self.graph_retriever,
                question=normalized_question,
                top_k=self.graph_top_k,
                max_depth=self.graph_max_depth,
            )
        hybrid_result = build_hybrid_retrieval_result(
            query=normalized_question,
            chunks=chunks,
            graph_relations=graph_relations,
        )
        prompt = build_hybrid_rag_prompt(
            question=normalized_question,
            hybrid_evidences=hybrid_result.evidences,
        )
        answer = self.llm_client.generate(prompt)
        sources = [
            SourceEvidence(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source=chunk.source,
                file_name=chunk.file_name,
                score=chunk.score,
                content=chunk.content,
            )
            for chunk in chunks
        ]
        graph_sources = [
            GraphEvidence(
                center_name=relation.center_name,
                center_type=relation.center_type,
                source_name=relation.source_name,
                source_type=relation.source_type,
                relation_type=relation.relation_type,
                target_name=relation.target_name,
                target_type=relation.target_type,
                chunk_id=relation.chunk_id,
                document_id=relation.document_id,
                source=relation.source,
                evidence=relation.evidence,
                confidence=relation.confidence,
            )
            for relation in graph_relations
        ]
        return QAResult(
            question=normalized_question,
            answer=answer,
            sources=sources,
            graph_sources=graph_sources,
        )


def retrieve_graph_relations_for_question(
    graph_retriever: GraphRelationRetriever,
    question: str,
    top_k: int,
    max_depth: int,
) -> list[RetrievedGraphRelation]:
    relations: list[RetrievedGraphRelation] = []
    for query in build_graph_query_terms(question):
        relations.extend(graph_retriever.retrieve(query=query, top_k=top_k, max_depth=max_depth))
    return deduplicate_graph_relations(relations)[:top_k]


def build_graph_query_terms(question: str) -> list[str]:
    query_terms = [question.strip()]
    query_terms.extend(extract_query_entities(question))
    return deduplicate_query_terms(query_terms)


def deduplicate_query_terms(query_terms: list[str]) -> list[str]:
    seen = set()
    deduplicated = []
    for query_term in query_terms:
        normalized_query_term = query_term.strip()
        key = normalized_query_term.lower()
        if normalized_query_term and key not in seen:
            seen.add(key)
            deduplicated.append(normalized_query_term)
    return deduplicated


def deduplicate_graph_relations(relations: list[RetrievedGraphRelation]) -> list[RetrievedGraphRelation]:
    seen = set()
    deduplicated = []
    for relation in relations:
        key = (
            relation.source_id,
            relation.relation_type,
            relation.target_id,
            relation.chunk_id,
            relation.evidence,
        )
        if key not in seen:
            seen.add(key)
            deduplicated.append(relation)
    return deduplicated
