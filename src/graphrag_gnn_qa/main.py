from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from graphrag_gnn_qa.api.routes_debug import router as debug_router
from graphrag_gnn_qa.api.routes_graph import router as graph_router
from graphrag_gnn_qa.api.routes_health import router as health_router
from graphrag_gnn_qa.api.routes_qa import router as qa_router
from graphrag_gnn_qa.api.routes_retrieve import router as retrieve_router
from graphrag_gnn_qa.config import Settings, get_settings
from graphrag_gnn_qa.runtime import RuntimeResources, build_runtime_resources


RuntimeFactory = Callable[[Settings], RuntimeResources]


def create_app(runtime_factory: RuntimeFactory = build_runtime_resources) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resources = runtime_factory(settings)
        app.state.runtime_resources = resources
        try:
            yield
        finally:
            try:
                resources.close()
            finally:
                del app.state.runtime_resources

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="A GraphRAG and GNN based QA system for complex relational knowledge reasoning.",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(retrieve_router)
    app.include_router(graph_router)
    app.include_router(debug_router)
    app.include_router(qa_router)
    return app


app = create_app()
