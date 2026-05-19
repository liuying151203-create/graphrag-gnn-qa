from typing import Any

import pytest

from graphrag_gnn_qa.retrieval.vector_retriever import RetrievedChunk, VectorRetriever


class FakeEmbeddingModel:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.query_embedding: list[float] | None = None
        self.top_k: int | None = None

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        self.query_embedding = query_embedding
        self.top_k = top_k
        return [
            {
                "score": 0.95,
                "chunk_id": "sample_chunk_0000",
                "document_id": "sample",
                "content": "GraphRAG connects vector search and graph traversal.",
                "source": "sample.txt",
                "file_name": "sample.txt",
                "file_type": "txt",
            }
        ]


def test_vector_retriever_returns_retrieved_chunks() -> None:
    vector_store = FakeVectorStore()
    retriever = VectorRetriever(embedding_model=FakeEmbeddingModel(), vector_store=vector_store)

    chunks = retriever.retrieve(query="What is GraphRAG?", top_k=3)

    assert vector_store.query_embedding == [17.0, 1.0]
    assert vector_store.top_k == 3
    assert chunks == [
        RetrievedChunk(
            score=0.95,
            chunk_id="sample_chunk_0000",
            document_id="sample",
            content="GraphRAG connects vector search and graph traversal.",
            source="sample.txt",
            file_name="sample.txt",
            file_type="txt",
        )
    ]


def test_vector_retriever_rejects_empty_query() -> None:
    retriever = VectorRetriever(embedding_model=FakeEmbeddingModel(), vector_store=FakeVectorStore())

    with pytest.raises(ValueError):
        retriever.retrieve(query="   ")


def test_vector_retriever_rejects_invalid_top_k() -> None:
    retriever = VectorRetriever(embedding_model=FakeEmbeddingModel(), vector_store=FakeVectorStore())

    with pytest.raises(ValueError):
        retriever.retrieve(query="GraphRAG", top_k=0)
