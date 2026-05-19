from dataclasses import dataclass
from typing import Protocol

from graphrag_gnn_qa.llm.client import LLMClient
from graphrag_gnn_qa.retrieval.vector_retriever import RetrievedChunk


class ChunkRetriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
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
class QAResult:
    question: str
    answer: str
    sources: list[SourceEvidence]


class RAGQAService:
    def __init__(self, retriever: ChunkRetriever, llm_client: LLMClient) -> None:
        self.retriever = retriever
        self.llm_client = llm_client

    def answer(self, question: str, top_k: int = 5) -> QAResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        chunks = self.retriever.retrieve(query=normalized_question, top_k=top_k)
        prompt = build_rag_prompt(question=normalized_question, chunks=chunks)
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
        return QAResult(question=normalized_question, answer=answer, sources=sources)


def build_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[Source {index}] chunk_id={chunk.chunk_id} score={chunk.score:.4f}\n{chunk.content}"
        for index, chunk in enumerate(chunks, start=1)
    )
    return (
        "Use the following retrieved context to answer the question.\n"
        "If the context does not contain enough information, say that the current documents do not provide enough evidence.\n"
        "Cite relevant source numbers in the answer when possible.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
