from dataclasses import dataclass
from time import perf_counter
from typing import Any

import httpx


class DemoApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class ApiResult:
    data: dict[str, Any]
    status_code: int
    latency_ms: float


class GraphRAGApiClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        normalized_base_url = base_url.strip().rstrip("/")
        if not normalized_base_url:
            raise ValueError("base_url must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        self._client = httpx.Client(
            base_url=normalized_base_url,
            timeout=timeout,
            transport=transport,
            trust_env=False,
        )

    def close(self) -> None:
        self._client.close()

    def health(self) -> ApiResult:
        return self._request("GET", "/health")

    def readiness(self) -> ApiResult:
        return self._request("GET", "/ready", accepted_statuses={200, 503})

    def retrieve(self, query: str, top_k: int) -> ApiResult:
        return self._request(
            "POST",
            "/retrieve",
            json={"query": query, "top_k": top_k},
        )

    def debug_retrieval(
        self,
        query: str,
        vector_top_k: int,
        graph_top_k: int,
        graph_max_depth: int,
    ) -> ApiResult:
        return self._request(
            "POST",
            "/retrieval/debug",
            json={
                "query": query,
                "vector_top_k": vector_top_k,
                "graph_top_k": graph_top_k,
                "graph_max_depth": graph_max_depth,
            },
        )

    def ask(self, question: str, top_k: int) -> ApiResult:
        return self._request(
            "POST",
            "/qa/ask",
            json={"question": question, "top_k": top_k},
        )

    def _request(
        self,
        method: str,
        path: str,
        accepted_statuses: set[int] | None = None,
        **kwargs,
    ) -> ApiResult:
        accepted_statuses = accepted_statuses or {200}
        started_at = perf_counter()
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.RequestError as exc:
            raise DemoApiError(f"Backend request failed: {exc.__class__.__name__}") from exc
        latency_ms = round((perf_counter() - started_at) * 1000, 3)
        data = _response_json(response)
        if response.status_code not in accepted_statuses:
            detail = data.get("detail") if isinstance(data, dict) else None
            message = str(detail or f"HTTP {response.status_code}")
            raise DemoApiError(f"Backend returned {response.status_code}: {message}")
        if not isinstance(data, dict):
            raise DemoApiError("Backend returned an invalid JSON object")
        return ApiResult(data=data, status_code=response.status_code, latency_ms=latency_ms)


def _response_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise DemoApiError("Backend returned a non-JSON response") from exc
