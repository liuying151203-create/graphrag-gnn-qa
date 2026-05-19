import argparse
import json
from pathlib import Path

from graphrag_gnn_qa.config import get_settings
from graphrag_gnn_qa.graph.neo4j_store import Neo4jGraphStore


def read_graph_records(input_file: Path) -> list[dict]:
    if not input_file.exists():
        raise FileNotFoundError(f"Graph triples file not found: {input_file}")

    return [
        json.loads(line)
        for line in input_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_graph_to_neo4j(input_file: Path, graph_store: Neo4jGraphStore, create_constraints: bool = True) -> int:
    records = read_graph_records(input_file)
    return graph_store.upsert_graph_records(records, create_constraints=create_constraints)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load graph entities and relations from JSONL into Neo4j.")
    parser.add_argument("--input-file", type=Path, default=Path("data/processed/graph_triples.jsonl"))
    parser.add_argument("--skip-constraints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    graph_store = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    try:
        loaded_count = load_graph_to_neo4j(
            input_file=args.input_file,
            graph_store=graph_store,
            create_constraints=not args.skip_constraints,
        )
    finally:
        graph_store.close()
    print(f"Loaded graph records from {loaded_count} chunks into Neo4j")


if __name__ == "__main__":
    main()
