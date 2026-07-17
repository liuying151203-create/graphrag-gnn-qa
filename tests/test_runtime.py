from fastapi.testclient import TestClient

from graphrag_gnn_qa.config import Settings
from graphrag_gnn_qa.main import create_app
from graphrag_gnn_qa.rag.qa_service import QAResult
from graphrag_gnn_qa.retrieval.vector_retriever import RetrievedChunk
from graphrag_gnn_qa.runtime import RuntimeResources, build_runtime_resources


class FakeVectorStore:
    def __init__(self, **kwargs) -> None:
        self.options = kwargs
        self.connect_count = 0
        self.close_count = 0

    def connect(self) -> None:
        self.connect_count += 1

    def close(self) -> None:
        self.close_count += 1

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        return []


class FakeGraphStore:
    def __init__(self, **kwargs) -> None:
        self.options = kwargs
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1

    def search_neighbors(self, query: str, top_k: int = 5, max_depth: int = 1) -> list[dict]:
        return []


class FakeEmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


class FakeVectorRetriever:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        self.calls += 1
        return [
            RetrievedChunk(
                score=0.9,
                chunk_id="sample_chunk_0000",
                document_id="sample",
                content="GraphRAG evidence.",
                source="sample.txt",
                file_name="sample.txt",
                file_type="txt",
            )
        ]


class FakeGraphRetriever:
    def retrieve(self, query: str, top_k: int = 5, max_depth: int = 1) -> list:
        return []


class FakeQAService:
    def answer(self, question: str, top_k: int = 5) -> QAResult:
        return QAResult(question=question, answer="answer", sources=[], graph_sources=[], citations=[])


class FakeLifecycleResources:
    def __init__(self) -> None:
        self.vector_retriever = FakeVectorRetriever()
        self.graph_retriever = FakeGraphRetriever()
        self.qa_service = FakeQAService()
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


def test_app_lifespan_reuses_resources_and_closes_them() -> None:
    resources = FakeLifecycleResources()
    factory_calls = []

    def runtime_factory(settings: Settings):
        factory_calls.append(settings)
        return resources

    test_app = create_app(runtime_factory=runtime_factory)

    with TestClient(test_app) as client:
        first_response = client.post("/retrieve", json={"query": "GraphRAG", "top_k": 1})
        second_response = client.post("/retrieve", json={"query": "GraphRAG", "top_k": 1})

        assert first_response.status_code == 200
        assert second_response.status_code == 200
        assert len(factory_calls) == 1
        assert resources.vector_retriever.calls == 2
        assert test_app.state.runtime_resources is resources

    assert resources.close_count == 1
    assert not hasattr(test_app.state, "runtime_resources")


def test_qa_endpoint_is_unavailable_when_runtime_has_no_qa_service() -> None:
    resources = FakeLifecycleResources()
    resources.qa_service = None
    test_app = create_app(runtime_factory=lambda settings: resources)

    with TestClient(test_app) as client:
        response = client.post("/qa/ask", json={"question": "What is GraphRAG?", "top_k": 1})

    assert response.status_code == 503
    assert response.json() == {"detail": "LLM_API_KEY is not configured"}


def test_build_runtime_resources_shares_retrievers_and_closes_stores(monkeypatch) -> None:
    monkeypatch.setattr(
        "graphrag_gnn_qa.runtime.SentenceTransformerEmbeddingModel",
        FakeEmbeddingModel,
    )
    monkeypatch.setattr("graphrag_gnn_qa.runtime.MilvusVectorStore", FakeVectorStore)
    monkeypatch.setattr("graphrag_gnn_qa.runtime.Neo4jGraphStore", FakeGraphStore)
    settings = Settings(llm_api_key="test-key", _env_file=None)

    resources = build_runtime_resources(settings)

    assert resources.vector_store.connect_count == 1
    assert resources.vector_retriever.vector_store is resources.vector_store
    assert resources.graph_retriever.graph_store is resources.graph_store
    assert resources.qa_service is not None
    assert resources.qa_service.retriever is resources.vector_retriever
    assert resources.qa_service.graph_retriever is resources.graph_retriever
    assert resources.qa_service.reranker is resources.reranker

    resources.close()

    assert resources.vector_store.close_count == 1
    assert resources.graph_store.close_count == 1


def test_build_runtime_resources_keeps_qa_unavailable_without_api_key(monkeypatch) -> None:
    monkeypatch.setattr(
        "graphrag_gnn_qa.runtime.SentenceTransformerEmbeddingModel",
        FakeEmbeddingModel,
    )
    monkeypatch.setattr("graphrag_gnn_qa.runtime.MilvusVectorStore", FakeVectorStore)
    monkeypatch.setattr("graphrag_gnn_qa.runtime.Neo4jGraphStore", FakeGraphStore)

    resources = build_runtime_resources(Settings(llm_api_key="", _env_file=None))

    assert resources.qa_service is None
    resources.close()
