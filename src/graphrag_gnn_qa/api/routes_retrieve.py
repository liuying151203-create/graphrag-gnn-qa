from typing import Protocol

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from graphrag_gnn_qa.api.dependencies import get_runtime_resources
from graphrag_gnn_qa.config import get_settings
from graphrag_gnn_qa.retrieval.vector_retriever import RetrievedChunk
from graphrag_gnn_qa.runtime import RuntimeResources


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


def get_vector_retriever(
    resources: RuntimeResources = Depends(get_runtime_resources),
) -> Retriever:
    return resources.vector_retriever


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
