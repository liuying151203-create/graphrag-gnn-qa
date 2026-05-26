import pytest

from graphrag_gnn_qa.rag.qa_service import (
    CitationEvidence,
    GraphEvidence,
    RAGQAService,
    SourceEvidence,
    build_graph_query_terms,
    build_rag_prompt,
    retrieve_graph_relations_for_question,
)
from graphrag_gnn_qa.retrieval.graph_retriever import RetrievedGraphRelation
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


class FakeGraphRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int = 5, max_depth: int = 1) -> list[RetrievedGraphRelation]:
        self.queries.append(query)
        return [
            RetrievedGraphRelation(
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
        ]


class EntityOnlyFakeGraphRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int = 5, max_depth: int = 1) -> list[RetrievedGraphRelation]:
        self.queries.append(query)
        if query != "GraphRAG":
            return []
        return FakeGraphRetriever().retrieve(query=query, top_k=top_k, max_depth=max_depth)


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
    assert "Graph Context" in prompt


def test_build_rag_prompt_contains_graph_context() -> None:
    graph_relations = FakeGraphRetriever().retrieve(query="GraphRAG")

    prompt = build_rag_prompt(question="What is GraphRAG?", chunks=[], graph_relations=graph_relations)

    assert "No vector context retrieved." in prompt
    assert "Graph Source 1" in prompt
    assert "GraphRAG (Method)" in prompt
    assert "SOLVES_TASK" in prompt


def test_rag_qa_service_returns_answer_and_sources() -> None:
    llm_client = FakeLLMClient()
    service = RAGQAService(retriever=FakeRetriever(), llm_client=llm_client)

    result = service.answer(question="What is GraphRAG?", top_k=3)

    assert result.question == "What is GraphRAG?"
    assert result.answer == "GraphRAG combines retrieved text chunks with generation."
    assert "What is GraphRAG?" in llm_client.prompt
    assert "Hybrid Evidence Context:" in llm_client.prompt
    assert "evidence_id=V1" in llm_client.prompt
    assert result.graph_sources == []
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
    assert result.citations == [
        CitationEvidence(
            evidence_id="V1",
            evidence_type="vector_chunk",
            document_id="sample",
            chunk_id="sample_chunk_0000",
            source="sample.txt",
            fusion_score=0.944,
        )
    ]


def test_rag_qa_service_returns_graph_sources() -> None:
    llm_client = FakeLLMClient()
    service = RAGQAService(
        retriever=FakeRetriever(),
        llm_client=llm_client,
        graph_retriever=FakeGraphRetriever(),
        graph_top_k=3,
        graph_max_depth=2,
    )

    result = service.answer(question="What is GraphRAG?", top_k=3)

    assert "Hybrid Evidence 1" in llm_client.prompt
    assert "evidence_id=V1+G1" in llm_client.prompt
    assert "type=hybrid" in llm_client.prompt
    assert "GraphRAG improves question answering." in llm_client.prompt
    assert result.citations == [
        CitationEvidence(
            evidence_id="V1+G1",
            evidence_type="hybrid",
            document_id="sample",
            chunk_id="sample_chunk_0000",
            source="sample.txt",
            fusion_score=0.944,
        )
    ]
    assert result.graph_sources == [
        GraphEvidence(
            center_name="GraphRAG",
            center_type="Method",
            source_name="GraphRAG",
            source_type="Method",
            relation_type="SOLVES_TASK",
            target_name="question answering",
            target_type="Task",
            chunk_id="sample_chunk_0000",
            document_id="sample",
            source="sample.txt",
            evidence="GraphRAG improves question answering.",
            confidence=0.9,
        )
    ]


def test_rag_qa_service_uses_custom_fusion_weights() -> None:
    service = RAGQAService(
        retriever=FakeRetriever(),
        llm_client=FakeLLMClient(),
        graph_retriever=FakeGraphRetriever(),
        fusion_score_weight=1,
        fusion_rank_weight=0,
    )

    result = service.answer(question="What is GraphRAG?", top_k=3)

    assert result.citations[0].fusion_score == 0.92


def test_build_graph_query_terms_extracts_entities_from_question() -> None:
    assert build_graph_query_terms("What is GraphRAG?") == ["What is GraphRAG?", "GraphRAG"]


def test_retrieve_graph_relations_for_question_uses_extracted_entity() -> None:
    graph_retriever = EntityOnlyFakeGraphRetriever()

    relations = retrieve_graph_relations_for_question(
        graph_retriever=graph_retriever,
        question="What is GraphRAG?",
        top_k=3,
        max_depth=2,
    )

    assert graph_retriever.queries[:2] == ["What is GraphRAG?", "GraphRAG"]
    assert len(relations) == 1
    assert relations[0].center_name == "GraphRAG"


def test_rag_qa_service_rejects_empty_question() -> None:
    service = RAGQAService(retriever=FakeRetriever(), llm_client=FakeLLMClient())

    with pytest.raises(ValueError):
        service.answer(question="   ")


def test_rag_qa_service_rejects_invalid_top_k() -> None:
    service = RAGQAService(retriever=FakeRetriever(), llm_client=FakeLLMClient())

    with pytest.raises(ValueError):
        service.answer(question="GraphRAG", top_k=0)
