from typing import Protocol

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from graphrag_gnn_qa.api.dependencies import get_runtime_resources
from graphrag_gnn_qa.config import get_settings
from graphrag_gnn_qa.rag.qa_service import QAResult
from graphrag_gnn_qa.runtime import RuntimeResources


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


class CitationEvidenceResponse(BaseModel):
    evidence_id: str
    evidence_type: str
    document_id: str
    chunk_id: str
    source: str
    fusion_score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceEvidenceResponse]
    graph_sources: list[GraphEvidenceResponse]
    citations: list[CitationEvidenceResponse]


class QAService(Protocol):
    def answer(self, question: str, top_k: int = 5) -> QAResult:
        ...


router = APIRouter(tags=["qa"])


def get_qa_service(
    resources: RuntimeResources = Depends(get_runtime_resources),
) -> QAService:
    if resources.qa_service is None:
        raise HTTPException(status_code=503, detail="LLM_API_KEY is not configured")
    return resources.qa_service


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
        citations=[CitationEvidenceResponse(**citation.__dict__) for citation in result.citations],
    )
