import json
from pathlib import Path

from scripts.load_graph_to_neo4j import load_graph_to_neo4j, read_graph_records


class FakeGraphStore:
    def __init__(self) -> None:
        self.records = []
        self.create_constraints = None

    def upsert_graph_records(self, records: list[dict], create_constraints: bool = True) -> int:
        self.records = records
        self.create_constraints = create_constraints
        return len(records)


def test_read_graph_records(tmp_path: Path) -> None:
    input_file = tmp_path / "graph_triples.jsonl"
    record = {
        "chunk_id": "sample_chunk_0000",
        "document_id": "sample",
        "source": "sample.txt",
        "entities": [],
        "relations": [],
    }
    input_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    records = read_graph_records(input_file)

    assert records == [record]


def test_load_graph_to_neo4j_calls_store(tmp_path: Path) -> None:
    input_file = tmp_path / "graph_triples.jsonl"
    record = {
        "chunk_id": "sample_chunk_0000",
        "document_id": "sample",
        "source": "sample.txt",
        "entities": [
            {"name": "GraphRAG", "type": "Method", "description": "Graph RAG method"},
        ],
        "relations": [],
    }
    input_file.write_text(json.dumps(record), encoding="utf-8")
    graph_store = FakeGraphStore()

    loaded_count = load_graph_to_neo4j(input_file=input_file, graph_store=graph_store, create_constraints=False)

    assert loaded_count == 1
    assert graph_store.records == [record]
    assert graph_store.create_constraints is False
