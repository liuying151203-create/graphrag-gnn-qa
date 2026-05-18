import pytest

from graphrag_gnn_qa.vectorstore.embedding import HashEmbeddingModel


def test_hash_embedding_model_returns_expected_dimension() -> None:
    model = HashEmbeddingModel(dimension=8)

    embeddings = model.embed_texts(["GraphRAG", "GNN"])

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 8
    assert len(embeddings[1]) == 8


def test_hash_embedding_model_returns_empty_list_for_empty_input() -> None:
    model = HashEmbeddingModel(dimension=8)

    embeddings = model.embed_texts([])

    assert embeddings == []


def test_hash_embedding_model_rejects_invalid_dimension() -> None:
    with pytest.raises(ValueError):
        HashEmbeddingModel(dimension=0)
