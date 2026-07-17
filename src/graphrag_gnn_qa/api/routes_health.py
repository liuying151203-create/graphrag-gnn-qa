from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel

from graphrag_gnn_qa.api.dependencies import get_runtime_resources
from graphrag_gnn_qa.config import get_settings
from graphrag_gnn_qa.runtime import (
    ComponentReadinessStatus,
    RuntimeReadinessStatus,
    RuntimeResources,
)

router = APIRouter(tags=["health"])


class ComponentReadinessResponse(BaseModel):
    status: ComponentReadinessStatus
    detail: str


class ReadinessResponse(BaseModel):
    status: RuntimeReadinessStatus
    components: dict[str, ComponentReadinessResponse]


@router.get("/health")
def health_check() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "environment": settings.app_env,
    }


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
def readiness_check(
    response: Response,
    resources: RuntimeResources = Depends(get_runtime_resources),
) -> ReadinessResponse:
    readiness = resources.readiness()
    if readiness.status != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status=readiness.status,
        components={
            name: ComponentReadinessResponse(status=component.status, detail=component.detail)
            for name, component in readiness.components.items()
        },
    )
