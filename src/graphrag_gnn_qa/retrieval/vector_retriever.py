from dataclasses import dataclass
from typing import Any, Protocol

from graphrag_gnn_qa.vectorstore.embedding import EmbeddingModel


class VectorSearchStore(Protocol):
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        ...


@dataclass(frozen=True)
class RetrievedChunk:
    score: float
    chunk_id: str
    document_id: str
    content: str
    source: str
    file_name: str
    file_type: str


class VectorRetriever:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorSearchStore) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_embedding = self.embedding_model.embed_texts([normalized_query])[0]
        results = self.vector_store.search(query_embedding=query_embedding, top_k=top_k)
        return [
            RetrievedChunk(
                score=float(result["score"]),
                chunk_id=result["chunk_id"],
                document_id=result["document_id"],
                content=result["content"],
                source=result["source"],
                file_name=result["file_name"],
                file_type=result["file_type"],
            )
            for result in results
        ]
