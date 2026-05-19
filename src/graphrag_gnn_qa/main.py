from fastapi import FastAPI

from graphrag_gnn_qa.api.routes_graph import router as graph_router
from graphrag_gnn_qa.api.routes_health import router as health_router
from graphrag_gnn_qa.api.routes_qa import router as qa_router
from graphrag_gnn_qa.api.routes_retrieve import router as retrieve_router
from graphrag_gnn_qa.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="A GraphRAG and GNN based QA system for complex relational knowledge reasoning.",
    )
    app.include_router(health_router)
    app.include_router(retrieve_router)
    app.include_router(graph_router)
    app.include_router(qa_router)
    return app


app = create_app()
