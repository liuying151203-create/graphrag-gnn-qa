from typing import Protocol

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from graphrag_gnn_qa.config import get_settings
from graphrag_gnn_qa.retrieval.vector_retriever import RetrievedChunk, VectorRetriever
from graphrag_gnn_qa.vectorstore.embedding import SentenceTransformerEmbeddingModel
from graphrag_gnn_qa.vectorstore.milvus_client import MilvusVectorStore


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)


class RetrievedChunkResponse(BaseModel):
    score: float
    chunk_id: str
    document_id: str
    content: str
    source: str
    file_name: str
    file_type: str


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    results: list[RetrievedChunkResponse]


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        ...


router = APIRouter(tags=["retrieval"])


def get_vector_retriever() -> Retriever:
    settings = get_settings()
    embedding_model = SentenceTransformerEmbeddingModel(model_name=settings.embedding_model)
    vector_store = MilvusVectorStore(
        host=settings.milvus_host,
        port=settings.milvus_port,
        collection_name=settings.milvus_chunk_collection,
    )
    vector_store.connect()
    return VectorRetriever(embedding_model=embedding_model, vector_store=vector_store)


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_chunks(
    request: RetrieveRequest,
    retriever: Retriever = Depends(get_vector_retriever),
) -> RetrieveResponse:
    settings = get_settings()
    top_k = request.top_k or settings.vector_top_k
    chunks = retriever.retrieve(query=request.query, top_k=top_k)
    return RetrieveResponse(
        query=request.query,
        top_k=top_k,
        results=[RetrievedChunkResponse(**chunk.__dict__) for chunk in chunks],
    )
