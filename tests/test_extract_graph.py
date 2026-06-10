import json
from pathlib import Path

from graphrag_gnn_qa.graph.extractor import GraphExtractor
from scripts.extract_graph import extract_graph, print_progress, read_chunk_records


class FakeLLMClient:
    def generate(self, prompt: str) -> str:
        return json.dumps(
            {
                "entities": [
                    {
                        "name": "GraphRAG",
                        "type": "Method",
                        "description": "Graph-based retrieval augmented generation",
                    }
                ],
                "relations": [],
            }
        )


class InvalidJsonLLMClient:
    def generate(self, prompt: str) -> str:
        return '{"entities": [{"name": "Broken", "type": "Concept", "description": "bad "quote""}], "relations": []}'


class FlakyLLMClient:
    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        if self.call_count == 1:
            raise TimeoutError("temporary timeout")
        return json.dumps(
            {
                "entities": [{"name": "Recovered", "type": "Concept", "description": ""}],
                "relations": [],
            }
        )


def test_read_chunk_records(tmp_path: Path) -> None:
    input_file = tmp_path / "chunks.jsonl"
    input_file.write_text(
        json.dumps({"chunk_id": "chunk_1", "content": "GraphRAG"}) + "\n",
        encoding="utf-8",
    )

    records = read_chunk_records(input_file)

    assert records == [{"chunk_id": "chunk_1", "content": "GraphRAG"}]


def test_extract_graph_writes_jsonl(tmp_path: Path) -> None:
    input_file = tmp_path / "chunks.jsonl"
    output_file = tmp_path / "graph_triples.jsonl"
    chunk = {
        "chunk_id": "sample_chunk_0000",
        "document_id": "sample",
        "source": "sample.txt",
        "content": "GraphRAG improves retrieval.",
    }
    input_file.write_text(json.dumps(chunk), encoding="utf-8")

    summary = extract_graph(
        input_file=input_file,
        output_file=output_file,
        extractor=GraphExtractor(llm_client=FakeLLMClient()),
    )

    records = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]

    assert summary["new_extracted_chunks"] == 1
    assert summary["total_successful_chunks"] == 1
    assert records[0]["chunk_id"] == "sample_chunk_0000"
    assert records[0]["entities"][0]["name"] == "GraphRAG"
    assert records[0]["entities"][0]["type"] == "Method"


def test_extract_graph_records_errors_and_continues(tmp_path: Path) -> None:
    input_file = tmp_path / "chunks.jsonl"
    output_file = tmp_path / "graph_triples.jsonl"
    error_file = tmp_path / "graph_extraction_errors.jsonl"
    chunk = {
        "chunk_id": "broken_chunk_0000",
        "document_id": "broken",
        "source": "broken.txt",
        "content": "This chunk produces invalid JSON.",
    }
    input_file.write_text(json.dumps(chunk), encoding="utf-8")

    summary = extract_graph(
        input_file=input_file,
        output_file=output_file,
        error_file=error_file,
        extractor=GraphExtractor(llm_client=InvalidJsonLLMClient()),
    )

    error_records = [json.loads(line) for line in error_file.read_text(encoding="utf-8").splitlines()]

    assert summary["new_extracted_chunks"] == 0
    assert summary["failed_chunks"] == 1
    assert output_file.read_text(encoding="utf-8") == ""
    assert error_records[0]["chunk_id"] == "broken_chunk_0000"
    assert error_records[0]["error_type"] == "JSONDecodeError"


def test_extract_graph_resume_skips_existing_successful_chunks(tmp_path: Path) -> None:
    input_file = tmp_path / "chunks.jsonl"
    output_file = tmp_path / "graph_triples.jsonl"
    existing_record = {
        "chunk_id": "existing_chunk_0000",
        "document_id": "sample",
        "source": "sample.txt",
        "entities": [],
        "relations": [],
    }
    new_chunk = {
        "chunk_id": "new_chunk_0001",
        "document_id": "sample",
        "source": "sample.txt",
        "content": "GraphRAG improves retrieval.",
    }
    input_file.write_text(
        json.dumps({**new_chunk, "chunk_id": "existing_chunk_0000"}) + "\n" + json.dumps(new_chunk),
        encoding="utf-8",
    )
    output_file.write_text(json.dumps(existing_record) + "\n", encoding="utf-8")

    summary = extract_graph(
        input_file=input_file,
        output_file=output_file,
        extractor=GraphExtractor(llm_client=FakeLLMClient()),
        resume=True,
        progress_every=0,
    )
    records = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]

    assert summary["existing_successful_chunks"] == 1
    assert summary["new_extracted_chunks"] == 1
    assert summary["total_successful_chunks"] == 2
    assert [record["chunk_id"] for record in records] == ["existing_chunk_0000", "new_chunk_0001"]


def test_extract_graph_retries_failed_chunk(tmp_path: Path) -> None:
    input_file = tmp_path / "chunks.jsonl"
    output_file = tmp_path / "graph_triples.jsonl"
    chunk = {
        "chunk_id": "flaky_chunk_0000",
        "document_id": "sample",
        "source": "sample.txt",
        "content": "Temporary network failures should be retried.",
    }
    input_file.write_text(json.dumps(chunk), encoding="utf-8")
    llm_client = FlakyLLMClient()

    summary = extract_graph(
        input_file=input_file,
        output_file=output_file,
        extractor=GraphExtractor(llm_client=llm_client),
        max_retries=1,
        retry_delay=0,
        progress_every=0,
    )
    records = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]

    assert llm_client.call_count == 2
    assert summary["new_extracted_chunks"] == 1
    assert summary["failed_chunks"] == 0
    assert records[0]["entities"][0]["name"] == "Recovered"


def test_print_progress_outputs_at_configured_interval(capsys) -> None:
    print_progress(
        index=2,
        total_count=3,
        existing_count=0,
        new_extracted_count=1,
        error_count=1,
        chunk={"chunk_id": "chunk_2"},
        progress_every=2,
    )

    assert "[extract_graph] 2/3 status=ok existing=0 new_ok=1 errors=1 chunk_id=chunk_2" in capsys.readouterr().out
