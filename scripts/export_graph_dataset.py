import argparse
import json
from pathlib import Path

from graphrag_gnn_qa.config import get_settings
from graphrag_gnn_qa.graph.neo4j_store import Neo4jGraphStore
from graphrag_gnn_qa.gnn.graph_dataset import GraphDataset


def write_graph_dataset(dataset: GraphDataset, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(dataset.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_graph_dataset(output_file: Path, graph_store: Neo4jGraphStore) -> GraphDataset:
    dataset = graph_store.export_graph_dataset()
    write_graph_dataset(dataset=dataset, output_file=output_file)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Neo4j graph nodes and edges for GNN dataset construction.")
    parser.add_argument("--output-file", type=Path, default=Path("data/processed/graph_dataset.json"))
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
        dataset = export_graph_dataset(output_file=args.output_file, graph_store=graph_store)
    finally:
        graph_store.close()
    print(
        f"Exported graph dataset with {len(dataset.nodes)} nodes and {len(dataset.edges)} edges: {args.output_file}"
    )


if __name__ == "__main__":
    main()
