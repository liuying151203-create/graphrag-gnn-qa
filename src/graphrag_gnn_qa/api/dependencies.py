from fastapi import HTTPException, Request

from graphrag_gnn_qa.runtime import RuntimeResources


def get_runtime_resources(request: Request) -> RuntimeResources:
    resources = getattr(request.app.state, "runtime_resources", None)
    if resources is None:
        raise HTTPException(status_code=503, detail="Application runtime resources are not initialized")
    return resources
