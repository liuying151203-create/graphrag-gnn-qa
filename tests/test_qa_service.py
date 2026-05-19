import pytest

from graphrag_gnn_qa.rag.qa_service import RAGQAService, SourceEvidence, build_rag_prompt
from graphrag_gnn_qa.retrieval.vector_retriever import RetrievedChunk


class FakeRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                score=0.92,
                chunk_id="sample_chunk_0000",
                document_id="sample",
                content="GraphRAG connects vector search and graph traversal.",
                source="sample.txt",
                file_name="sample.txt",
                file_type="txt",
            )
        ]


class FakeLLMClient:
    def __init__(self) -> None:
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return "GraphRAG combines retrieved text chunks with generation."


def test_build_rag_prompt_contains_question_and_context() -> None:
    chunks = [
        RetrievedChunk(
            score=0.92,
            chunk_id="sample_chunk_0000",
            document_id="sample",
            content="GraphRAG connects vector search and graph traversal.",
            source="sample.txt",
            file_name="sample.txt",
            file_type="txt",
        )
    ]

    prompt = build_rag_prompt(question="What is GraphRAG?", chunks=chunks)

    assert "What is GraphRAG?" in prompt
    assert "GraphRAG connects vector search and graph traversal." in prompt
    assert "Source 1" in prompt


def test_rag_qa_service_returns_answer_and_sources() -> None:
    llm_client = FakeLLMClient()
    service = RAGQAService(retriever=FakeRetriever(), llm_client=llm_client)

    result = service.answer(question="What is GraphRAG?", top_k=3)

    assert result.question == "What is GraphRAG?"
    assert result.answer == "GraphRAG combines retrieved text chunks with generation."
    assert "What is GraphRAG?" in llm_client.prompt
    assert result.sources == [
        SourceEvidence(
            chunk_id="sample_chunk_0000",
            document_id="sample",
            source="sample.txt",
            file_name="sample.txt",
            score=0.92,
            content="GraphRAG connects vector search and graph traversal.",
        )
    ]


def test_rag_qa_service_rejects_empty_question() -> None:
    service = RAGQAService(retriever=FakeRetriever(), llm_client=FakeLLMClient())

    with pytest.raises(ValueError):
        service.answer(question="   ")


def test_rag_qa_service_rejects_invalid_top_k() -> None:
    service = RAGQAService(retriever=FakeRetriever(), llm_client=FakeLLMClient())

    with pytest.raises(ValueError):
        service.answer(question="GraphRAG", top_k=0)
