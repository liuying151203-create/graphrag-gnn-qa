import argparse

from graphrag_gnn_qa.config import get_settings
from graphrag_gnn_qa.graph.neo4j_store import Neo4jGraphStore
from graphrag_gnn_qa.retrieval.graph_retriever import GraphRetriever


def search_graph(query: str, top_k: int, max_depth: int) -> list[dict]:
    settings = get_settings()
    graph_store = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    try:
        retriever = GraphRetriever(graph_store=graph_store)
        relations = retriever.retrieve(query=query, top_k=top_k, max_depth=max_depth)
        return [relation.__dict__ for relation in relations]
    finally:
        graph_store.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search graph neighbors from Neo4j by query text.")
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    top_k = args.top_k or settings.graph_top_k
    max_depth = args.max_depth or settings.graph_max_depth
    results = search_graph(query=args.query, top_k=top_k, max_depth=max_depth)

    for index, result in enumerate(results, start=1):
        print(
            f"[{index}] {result['source_name']} "
            f"-[:{result['relation_type']}]-> "
            f"{result['target_name']}"
        )
        print(f"center={result['center_name']} depth<={max_depth} confidence={result['confidence']:.4f}")
        print(f"chunk_id={result['chunk_id']} source={result['source']}")
        print(f"evidence={result['evidence']}")
        print()


if __name__ == "__main__":
    main()
