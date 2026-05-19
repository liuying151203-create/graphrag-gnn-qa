import pytest

from graphrag_gnn_qa.retrieval.graph_retriever import GraphRetriever, RetrievedGraphRelation


class FakeGraphStore:
    def search_neighbors(self, query: str, top_k: int = 5, max_depth: int = 1) -> list[dict]:
        return [
            {
                "center_id": "Method:graphrag",
                "center_name": "GraphRAG",
                "center_type": "Method",
                "source_id": "Method:graphrag",
                "source_name": "GraphRAG",
                "source_type": "Method",
                "relation_type": "SOLVES_TASK",
                "target_id": "Task:question answering",
                "target_name": "question answering",
                "target_type": "Task",
                "chunk_id": "sample_chunk_0000",
                "document_id": "sample",
                "source": "sample.txt",
                "evidence": "GraphRAG improves question answering.",
                "confidence": 0.9,
            }
        ]


def test_graph_retriever_returns_relations() -> None:
    retriever = GraphRetriever(graph_store=FakeGraphStore())

    results = retriever.retrieve(query="GraphRAG", top_k=3, max_depth=1)

    assert results == [
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


def test_graph_retriever_rejects_empty_query() -> None:
    retriever = GraphRetriever(graph_store=FakeGraphStore())

    with pytest.raises(ValueError):
        retriever.retrieve(query="   ")


def test_graph_retriever_rejects_invalid_top_k() -> None:
    retriever = GraphRetriever(graph_store=FakeGraphStore())

    with pytest.raises(ValueError):
        retriever.retrieve(query="GraphRAG", top_k=0)


def test_graph_retriever_rejects_invalid_max_depth() -> None:
    retriever = GraphRetriever(graph_store=FakeGraphStore())

    with pytest.raises(ValueError):
        retriever.retrieve(query="GraphRAG", max_depth=0)
