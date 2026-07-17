from time import perf_counter

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
from graphrag_gnn_qa.retrieval.hybrid_result import build_hybrid_retrieval_result


class RetrievalDebugRequest(BaseModel):
    query: str = Field(min_length=1)
    vector_top_k: int | None = Field(default=None, ge=1)
    graph_top_k: int | None = Field(default=None, ge=1)
    graph_max_depth: int | None = Field(default=None, ge=1)


class HybridEvidenceResponse(BaseModel):
    evidence_id: str
    evidence_type: str
    rank: int
    score: float
    fusion_score: float
    document_id: str
    chunk_id: str
    source: str
    content: str
    metadata: dict[str, str]


class FusionWeightsResponse(BaseModel):
    score_weight: float
    rank_weight: float


class RetrievalTimingsResponse(BaseModel):
    vector_ms: float
    graph_ms: float
    fusion_ms: float
    total_ms: float


class RetrievalDebugResponse(BaseModel):
    query: str
    vector_top_k: int
    graph_top_k: int
    graph_max_depth: int
    fusion_weights: FusionWeightsResponse
    timings: RetrievalTimingsResponse
    graph_query_terms: list[str]
    vector_results: list[RetrievedChunkResponse]
    graph_results: list[RetrievedGraphRelationResponse]
    hybrid_results: list[HybridEvidenceResponse]


router = APIRouter(tags=["retrieval"])


@router.post("/retrieval/debug", response_model=RetrievalDebugResponse)
def debug_retrieval(
    request: RetrievalDebugRequest,
    vector_retriever: Retriever = Depends(get_vector_retriever),
    graph_retriever: GraphRelationRetriever = Depends(get_graph_retriever),
) -> RetrievalDebugResponse:
    total_started_at = perf_counter()
    settings = get_settings()
    vector_top_k = request.vector_top_k or settings.vector_top_k
    graph_top_k = request.graph_top_k or settings.graph_top_k
    graph_max_depth = request.graph_max_depth or settings.graph_max_depth

    vector_started_at = perf_counter()
    vector_results = vector_retriever.retrieve(query=request.query, top_k=vector_top_k)
    vector_ms = _elapsed_ms(vector_started_at)

    graph_started_at = perf_counter()
    graph_query_terms = build_graph_query_terms(request.query)
    graph_results = retrieve_graph_relations_for_question(
        graph_retriever=graph_retriever,
        question=request.query,
        top_k=graph_top_k,
        max_depth=graph_max_depth,
    )
    graph_ms = _elapsed_ms(graph_started_at)

    fusion_started_at = perf_counter()
    hybrid_result = build_hybrid_retrieval_result(
        query=request.query,
        chunks=vector_results,
        graph_relations=graph_results,
        score_weight=settings.fusion_score_weight,
        rank_weight=settings.fusion_rank_weight,
    )
    fusion_ms = _elapsed_ms(fusion_started_at)

    return RetrievalDebugResponse(
        query=request.query,
        vector_top_k=vector_top_k,
        graph_top_k=graph_top_k,
        graph_max_depth=graph_max_depth,
        fusion_weights=FusionWeightsResponse(
            score_weight=settings.fusion_score_weight,
            rank_weight=settings.fusion_rank_weight,
        ),
        timings=RetrievalTimingsResponse(
            vector_ms=vector_ms,
            graph_ms=graph_ms,
            fusion_ms=fusion_ms,
            total_ms=_elapsed_ms(total_started_at),
        ),
        graph_query_terms=graph_query_terms,
        vector_results=[RetrievedChunkResponse(**chunk.__dict__) for chunk in vector_results],
        graph_results=[RetrievedGraphRelationResponse(**relation.__dict__) for relation in graph_results],
        hybrid_results=[
            HybridEvidenceResponse(
                evidence_id=evidence.evidence_id,
                evidence_type=evidence.evidence_type.value,
                rank=evidence.rank,
                score=evidence.score,
                fusion_score=evidence.fusion_score,
                document_id=evidence.document_id,
                chunk_id=evidence.chunk_id,
                source=evidence.source,
                content=evidence.content,
                metadata=evidence.metadata,
            )
            for evidence in hybrid_result.evidences
        ],
    )


def _elapsed_ms(started_at: float) -> float:
    return round(max(0.0, (perf_counter() - started_at) * 1000), 3)
