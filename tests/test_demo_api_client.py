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


def test_demo_api_client_uploads_and_deletes_document() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(
                201,
                json={"status": "completed", "document_id": "doc_123"},
            )
        return httpx.Response(
            200,
            json={"status": "completed", "document_id": "doc_123"},
        )

    client = GraphRAGApiClient(transport=httpx.MockTransport(handler))

    upload = client.upload_document(
        filename="paper.txt",
        content=b"GraphRAG content",
        content_type="text/plain",
        overwrite=True,
    )
    deletion = client.delete_document("doc_123")
    client.close()

    upload_body = requests[0].read()
    assert upload.status_code == 201
    assert b'name="overwrite"' in upload_body
    assert b"true" in upload_body
    assert b'filename="paper.txt"' in upload_body
    assert b"GraphRAG content" in upload_body
    assert requests[1].method == "DELETE"
    assert requests[1].url.path == "/documents/doc_123"
    assert deletion.status_code == 200


def test_demo_api_client_queues_and_queries_ingestion_task() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(
                202,
                json={"status": "pending", "task_id": "ing_123"},
            )
        return httpx.Response(
            200,
            json={"status": "processing", "task_id": "ing_123", "progress": 55},
        )

    client = GraphRAGApiClient(transport=httpx.MockTransport(handler))

    queued = client.queue_document_upload(
        filename="paper.txt",
        content=b"GraphRAG content",
        content_type="text/plain",
        overwrite=True,
    )
    task = client.get_ingestion_task("ing_123")
    client.close()

    upload_body = requests[0].read()
    assert queued.status_code == 202
    assert requests[0].url.path == "/documents/upload/async"
    assert b'name="overwrite"' in upload_body
    assert b"true" in upload_body
    assert requests[1].method == "GET"
    assert requests[1].url.path == "/documents/tasks/ing_123"
    assert task.data["progress"] == 55
