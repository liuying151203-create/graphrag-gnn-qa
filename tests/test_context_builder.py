from graphrag_gnn_qa.rag.context_builder import (
    GRAPH_EMPTY_CONTEXT,
    HYBRID_EMPTY_CONTEXT,
    VECTOR_EMPTY_CONTEXT,
    build_graph_context,
    build_graphrag_context,
    build_hybrid_context,
    build_hybrid_rag_prompt,
    build_rag_prompt,
    build_vector_context,
)
from graphrag_gnn_qa.retrieval.graph_retriever import RetrievedGraphRelation
from graphrag_gnn_qa.retrieval.hybrid_result import build_hybrid_evidences
from graphrag_gnn_qa.retrieval.vector_retriever import RetrievedChunk


def sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        score=0.92,
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


def test_build_vector_context_returns_empty_message_without_chunks() -> None:
    assert build_vector_context([]) == VECTOR_EMPTY_CONTEXT


def test_build_vector_context_formats_chunks() -> None:
    context = build_vector_context([sample_chunk()])

    assert "[Source 1]" in context
    assert "chunk_id=sample_chunk_0000" in context
    assert "score=0.9200" in context
    assert "GraphRAG connects vector search and graph traversal." in context


def test_build_graph_context_returns_empty_message_without_relations() -> None:
    assert build_graph_context([]) == GRAPH_EMPTY_CONTEXT
    assert build_graph_context() == GRAPH_EMPTY_CONTEXT


def test_build_graph_context_formats_relations() -> None:
    context = build_graph_context([sample_graph_relation()])

    assert "[Graph Source 1]" in context
    assert "center=GraphRAG (Method)" in context
    assert "-[:SOLVES_TASK]->" in context
    assert "question answering (Task)" in context
    assert "confidence=0.9000" in context
    assert "evidence=GraphRAG improves question answering." in context


def test_build_hybrid_context_returns_empty_message_without_evidence() -> None:
    assert build_hybrid_context([]) == HYBRID_EMPTY_CONTEXT


def test_build_hybrid_context_formats_hybrid_evidence() -> None:
    evidences = build_hybrid_evidences(chunks=[sample_chunk()], graph_relations=[sample_graph_relation()])

    context = build_hybrid_context(evidences)

    assert "[Hybrid Evidence 1]" in context
    assert "evidence_id=V1+G1" in context
    assert "type=hybrid" in context
    assert "fusion_score=" in context
    assert "GraphRAG connects vector search and graph traversal." in context
    assert "GraphRAG improves question answering." in context


def test_build_graphrag_context_combines_vector_and_graph_context() -> None:
    context = build_graphrag_context(chunks=[sample_chunk()], graph_relations=[sample_graph_relation()])

    assert "[Source 1]" in context.vector_context
    assert "[Graph Source 1]" in context.graph_context


def test_build_hybrid_rag_prompt_contains_hybrid_context() -> None:
    evidences = build_hybrid_evidences(chunks=[sample_chunk()], graph_relations=[sample_graph_relation()])

    prompt = build_hybrid_rag_prompt(question="What is GraphRAG?", hybrid_evidences=evidences)

    assert "Hybrid Evidence Context:" in prompt
    assert "Question:" in prompt
    assert "What is GraphRAG?" in prompt
    assert "[Hybrid Evidence 1]" in prompt
    assert "evidence_id=V1+G1" in prompt
    assert prompt.endswith("Answer:")


def test_build_rag_prompt_contains_all_context_sections() -> None:
    prompt = build_rag_prompt(
        question="What is GraphRAG?",
        chunks=[sample_chunk()],
        graph_relations=[sample_graph_relation()],
    )

    assert "Vector Context:" in prompt
    assert "Graph Context:" in prompt
    assert "Question:" in prompt
    assert "What is GraphRAG?" in prompt
    assert "[Source 1]" in prompt
    assert "[Graph Source 1]" in prompt
    assert prompt.endswith("Answer:")
