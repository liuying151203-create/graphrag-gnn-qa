from typing import Protocol

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from graphrag_gnn_qa.api.dependencies import get_runtime_resources
from graphrag_gnn_qa.config import get_settings
from graphrag_gnn_qa.retrieval.graph_retriever import RetrievedGraphRelation
from graphrag_gnn_qa.runtime import RuntimeResources


class GraphRetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    max_depth: int | None = Field(default=None, ge=1)


class RetrievedGraphRelationResponse(BaseModel):
    center_id: str
    center_name: str
    center_type: str
    source_id: str
    source_name: str
    source_type: str
    relation_type: str
    target_id: str
    target_name: str
    target_type: str
    chunk_id: str
    document_id: str
    source: str
    evidence: str
    confidence: float


class GraphRetrieveResponse(BaseModel):
    query: str
    top_k: int
    max_depth: int
    results: list[RetrievedGraphRelationResponse]


class GraphRelationRetriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5, max_depth: int = 1) -> list[RetrievedGraphRelation]:
        ...


router = APIRouter(tags=["graph"])


def get_graph_retriever(
    resources: RuntimeResources = Depends(get_runtime_resources),
) -> GraphRelationRetriever:
    return resources.graph_retriever


@router.post("/graph/retrieve", response_model=GraphRetrieveResponse)
def retrieve_graph(
    request: GraphRetrieveRequest,
    retriever: GraphRelationRetriever = Depends(get_graph_retriever),
) -> GraphRetrieveResponse:
    settings = get_settings()
    top_k = request.top_k or settings.graph_top_k
    max_depth = request.max_depth or settings.graph_max_depth
    relations = retriever.retrieve(query=request.query, top_k=top_k, max_depth=max_depth)
    return GraphRetrieveResponse(
        query=request.query,
        top_k=top_k,
        max_depth=max_depth,
        results=[RetrievedGraphRelationResponse(**relation.__dict__) for relation in relations],
    )
