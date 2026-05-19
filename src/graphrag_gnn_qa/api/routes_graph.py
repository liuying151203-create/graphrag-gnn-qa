from typing import Protocol

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from graphrag_gnn_qa.config import get_settings
from graphrag_gnn_qa.graph.neo4j_store import Neo4jGraphStore
from graphrag_gnn_qa.retrieval.graph_retriever import GraphRetriever, RetrievedGraphRelation


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


def get_graph_retriever() -> GraphRelationRetriever:
    settings = get_settings()
    graph_store = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    return GraphRetriever(graph_store=graph_store)


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
