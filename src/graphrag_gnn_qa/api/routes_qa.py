from typing import Protocol

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from graphrag_gnn_qa.config import get_settings
from graphrag_gnn_qa.graph.neo4j_store import Neo4jGraphStore
from graphrag_gnn_qa.llm.client import OpenAICompatibleLLMClient
from graphrag_gnn_qa.rag.qa_service import QAResult, RAGQAService
from graphrag_gnn_qa.retrieval.graph_retriever import GraphRetriever
from graphrag_gnn_qa.retrieval.vector_retriever import VectorRetriever
from graphrag_gnn_qa.vectorstore.embedding import SentenceTransformerEmbeddingModel
from graphrag_gnn_qa.vectorstore.milvus_client import MilvusVectorStore


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)


class SourceEvidenceResponse(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    file_name: str
    score: float
    content: str


class GraphEvidenceResponse(BaseModel):
    center_name: str
    center_type: str
    source_name: str
    source_type: str
    relation_type: str
    target_name: str
    target_type: str
    chunk_id: str
    document_id: str
    source: str
    evidence: str
    confidence: float


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceEvidenceResponse]
    graph_sources: list[GraphEvidenceResponse]


class QAService(Protocol):
    def answer(self, question: str, top_k: int = 5) -> QAResult:
        ...


router = APIRouter(tags=["qa"])


def get_qa_service() -> QAService:
    settings = get_settings()
    if not settings.llm_api_key:
        raise HTTPException(status_code=503, detail="LLM_API_KEY is not configured")

    try:
        embedding_model = SentenceTransformerEmbeddingModel(model_name=settings.embedding_model)
        vector_store = MilvusVectorStore(
            host=settings.milvus_host,
            port=settings.milvus_port,
            collection_name=settings.milvus_chunk_collection,
        )
        vector_store.connect()
        retriever = VectorRetriever(embedding_model=embedding_model, vector_store=vector_store)
        graph_store = Neo4jGraphStore(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        graph_retriever = GraphRetriever(graph_store=graph_store)
        llm_client = OpenAICompatibleLLMClient(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
        )
        return RAGQAService(
            retriever=retriever,
            llm_client=llm_client,
            graph_retriever=graph_retriever,
            graph_top_k=settings.graph_top_k,
            graph_max_depth=settings.graph_max_depth,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to initialize QA service: {exc}") from exc


@router.post("/qa/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    qa_service: QAService = Depends(get_qa_service),
) -> AskResponse:
    settings = get_settings()
    top_k = request.top_k or settings.vector_top_k
    try:
        result = qa_service.answer(question=request.question, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"LLM provider returned an error: {exc.response.text}") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Failed to request LLM provider: {exc}") from exc
    return AskResponse(
        question=result.question,
        answer=result.answer,
        sources=[SourceEvidenceResponse(**source.__dict__) for source in result.sources],
        graph_sources=[GraphEvidenceResponse(**source.__dict__) for source in result.graph_sources],
    )
