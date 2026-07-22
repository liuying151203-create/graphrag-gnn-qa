import httpx
import pytest

from demo.api_client import DemoApiError, GraphRAGApiClient


def test_demo_api_client_sends_retrieval_and_qa_payloads() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/retrieval/debug":
            return httpx.Response(200, json={"hybrid_results": [], "timings": {}})
        if request.url.path == "/qa/ask":
            return httpx.Response(200, json={"answer": "GraphRAG", "citations": [], "timings": {}})
        return httpx.Response(404, json={"detail": "not found"})

    client = GraphRAGApiClient(transport=httpx.MockTransport(handler))

    retrieval = client.debug_retrieval(
        query="What is GraphRAG?",
        vector_top_k=3,
        graph_top_k=5,
        graph_max_depth=2,
    )
    qa = client.ask(question="What is GraphRAG?", top_k=3)
    client.close()

    assert retrieval.status_code == 200
    assert retrieval.latency_ms >= 0
    assert qa.data["answer"] == "GraphRAG"
    assert requests[0].read().decode("utf-8") == (
        '{"query":"What is GraphRAG?","vector_top_k":3,"graph_top_k":5,"graph_max_depth":2}'
    )
    assert requests[1].read().decode("utf-8") == '{"question":"What is GraphRAG?","top_k":3}'


def test_demo_api_client_accepts_degraded_readiness_response() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            503,
            json={"status": "degraded", "components": {"llm": {"status": "not_configured"}}},
        )
    )
    client = GraphRAGApiClient(transport=transport)

    result = client.readiness()
    client.close()

    assert result.status_code == 503
    assert result.data["status"] == "degraded"


def test_demo_api_client_raises_sanitized_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret backend address", request=request)

    client = GraphRAGApiClient(transport=httpx.MockTransport(handler))

    with pytest.raises(DemoApiError, match="Backend request failed: ConnectError") as exc_info:
        client.health()

    client.close()
    assert "secret backend address" not in str(exc_info.value)


def test_demo_api_client_rejects_backend_error_response() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(502, json={"detail": "LLM provider unavailable"})
    )
    client = GraphRAGApiClient(transport=transport)

    with pytest.raises(DemoApiError, match="Backend returned 502: LLM provider unavailable"):
        client.ask(question="GraphRAG", top_k=3)

    client.close()
