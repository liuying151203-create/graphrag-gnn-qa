import pytest

from graphrag_gnn_qa.retrieval.graph_retriever import RetrievedGraphRelation
from graphrag_gnn_qa.retrieval.hybrid_result import (
    EvidenceType,
    build_hybrid_evidences,
    build_hybrid_retrieval_result,
    chunk_to_hybrid_evidence,
    graph_relation_to_hybrid_evidence,
)
from graphrag_gnn_qa.retrieval.vector_retriever import RetrievedChunk


def sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        score=0.91,
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


def test_chunk_to_hybrid_evidence() -> None:
    evidence = chunk_to_hybrid_evidence(chunk=sample_chunk(), rank=1)

    assert evidence.evidence_id == "V1"
    assert evidence.evidence_type == EvidenceType.VECTOR_CHUNK
    assert evidence.rank == 1
    assert evidence.score == 0.91
    assert evidence.document_id == "sample"
    assert evidence.chunk_id == "sample_chunk_0000"
    assert evidence.source == "sample.txt"
    assert evidence.content == "GraphRAG connects vector search and graph traversal."
    assert evidence.metadata == {"file_name": "sample.txt", "file_type": "txt"}


def test_graph_relation_to_hybrid_evidence() -> None:
    evidence = graph_relation_to_hybrid_evidence(relation=sample_graph_relation(), rank=1)

    assert evidence.evidence_id == "G1"
    assert evidence.evidence_type == EvidenceType.GRAPH_RELATION
    assert evidence.rank == 1
    assert evidence.score == 0.9
    assert evidence.document_id == "sample"
    assert evidence.chunk_id == "sample_chunk_0000"
    assert evidence.source == "sample.txt"
    assert evidence.content == "GraphRAG improves question answering."
    assert evidence.metadata["center_name"] == "GraphRAG"
    assert evidence.metadata["relation_type"] == "SOLVES_TASK"
    assert evidence.metadata["target_name"] == "question answering"


def test_build_hybrid_evidences_combines_vector_and_graph_evidence() -> None:
    evidences = build_hybrid_evidences(chunks=[sample_chunk()], graph_relations=[sample_graph_relation()])

    assert [evidence.evidence_id for evidence in evidences] == ["V1", "G1"]
    assert [evidence.evidence_type for evidence in evidences] == [EvidenceType.VECTOR_CHUNK, EvidenceType.GRAPH_RELATION]


def test_build_hybrid_retrieval_result() -> None:
    result = build_hybrid_retrieval_result(
        query="What is GraphRAG?",
        chunks=[sample_chunk()],
        graph_relations=[sample_graph_relation()],
    )

    assert result.query == "What is GraphRAG?"
    assert len(result.evidences) == 2


def test_build_hybrid_retrieval_result_rejects_empty_query() -> None:
    with pytest.raises(ValueError):
        build_hybrid_retrieval_result(query="   ", chunks=[], graph_relations=[])


def test_hybrid_evidence_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError):
        chunk_to_hybrid_evidence(chunk=sample_chunk(), rank=0)
    with pytest.raises(ValueError):
        graph_relation_to_hybrid_evidence(relation=sample_graph_relation(), rank=0)
