from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from graphrag_gnn_qa.api.routes_graph import (
    GraphRelationRetriever,
    RetrievedGraphRelationResponse,
    get_graph_retriever,
)
from graphrag_gnn_qa.api.routes_retrieve import Retriever, RetrievedChunkResponse, get_vector_retriever
from graphrag_gnn_qa.config import get_settings
from graphrag_gnn_qa.rag.qa_service import build_graph_query_terms, retrieve_graph_relations_for_question


class RetrievalDebugRequest(BaseModel):
    query: str = Field(min_length=1)
    vector_top_k: int | None = Field(default=None, ge=1)
    graph_top_k: int | None = Field(default=None, ge=1)
    graph_max_depth: int | None = Field(default=None, ge=1)


class RetrievalDebugResponse(BaseModel):
    query: str
    vector_top_k: int
    graph_top_k: int
    graph_max_depth: int
    graph_query_terms: list[str]
    vector_results: list[RetrievedChunkResponse]
    graph_results: list[RetrievedGraphRelationResponse]


router = APIRouter(tags=["retrieval"])


@router.post("/retrieval/debug", response_model=RetrievalDebugResponse)
def debug_retrieval(
    request: RetrievalDebugRequest,
    vector_retriever: Retriever = Depends(get_vector_retriever),
    graph_retriever: GraphRelationRetriever = Depends(get_graph_retriever),
) -> RetrievalDebugResponse:
    settings = get_settings()
    vector_top_k = request.vector_top_k or settings.vector_top_k
    graph_top_k = request.graph_top_k or settings.graph_top_k
    graph_max_depth = request.graph_max_depth or settings.graph_max_depth

    vector_results = vector_retriever.retrieve(query=request.query, top_k=vector_top_k)
    graph_query_terms = build_graph_query_terms(request.query)
    graph_results = retrieve_graph_relations_for_question(
        graph_retriever=graph_retriever,
        question=request.query,
        top_k=graph_top_k,
        max_depth=graph_max_depth,
    )

    return RetrievalDebugResponse(
        query=request.query,
        vector_top_k=vector_top_k,
        graph_top_k=graph_top_k,
        graph_max_depth=graph_max_depth,
        graph_query_terms=graph_query_terms,
        vector_results=[RetrievedChunkResponse(**chunk.__dict__) for chunk in vector_results],
        graph_results=[RetrievedGraphRelationResponse(**relation.__dict__) for relation in graph_results],
    )
