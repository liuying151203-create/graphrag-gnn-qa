from dataclasses import dataclass
from typing import Protocol

from graphrag_gnn_qa.llm.client import LLMClient
from graphrag_gnn_qa.retrieval.graph_retriever import RetrievedGraphRelation
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
            graph_relations = self.graph_retriever.retrieve(
                query=normalized_question,
                top_k=self.graph_top_k,
                max_depth=self.graph_max_depth,
            )
        prompt = build_rag_prompt(question=normalized_question, chunks=chunks, graph_relations=graph_relations)
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


def build_rag_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    graph_relations: list[RetrievedGraphRelation] | None = None,
) -> str:
    vector_context = "\n\n".join(
        f"[Source {index}] chunk_id={chunk.chunk_id} score={chunk.score:.4f}\n{chunk.content}"
        for index, chunk in enumerate(chunks, start=1)
    )
    if not vector_context:
        vector_context = "No vector context retrieved."

    graph_context = "\n\n".join(
        f"[Graph Source {index}] "
        f"center={relation.center_name} ({relation.center_type}) "
        f"triple={relation.source_name} ({relation.source_type}) "
        f"-[:{relation.relation_type}]-> "
        f"{relation.target_name} ({relation.target_type}) "
        f"chunk_id={relation.chunk_id} confidence={relation.confidence:.4f}\n"
        f"evidence={relation.evidence}"
        for index, relation in enumerate(graph_relations or [], start=1)
    )
    if not graph_context:
        graph_context = "No graph context retrieved."

    return (
        "Use the following retrieved vector context and graph context to answer the question.\n"
        "If the context does not contain enough information, say that the current documents do not provide enough evidence.\n"
        "Cite relevant vector source numbers and graph source numbers in the answer when possible.\n\n"
        f"Vector Context:\n{vector_context}\n\n"
        f"Graph Context:\n{graph_context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
