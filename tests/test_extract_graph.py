import json
from pathlib import Path

from graphrag_gnn_qa.graph.extractor import GraphExtractor
from scripts.extract_graph import extract_graph, read_chunk_records


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

    extracted_count = extract_graph(
        input_file=input_file,
        output_file=output_file,
        extractor=GraphExtractor(llm_client=FakeLLMClient()),
    )

    records = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]

    assert extracted_count == 1
    assert records[0]["chunk_id"] == "sample_chunk_0000"
    assert records[0]["entities"][0]["name"] == "GraphRAG"
    assert records[0]["entities"][0]["type"] == "Method"
