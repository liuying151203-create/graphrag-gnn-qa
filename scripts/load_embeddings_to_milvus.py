import argparse
from pathlib import Path

from graphrag_gnn_qa.config import get_settings
from graphrag_gnn_qa.vectorstore.milvus_client import (
    MilvusVectorStore,
    infer_embedding_dimension,
    read_embedding_records,
)


def load_embeddings_to_milvus(
    input_file: Path,
    collection_name: str,
    host: str,
    port: int,
    drop_existing: bool = False,
) -> int:
    records = read_embedding_records(input_file)
    dimension = infer_embedding_dimension(records)
    vector_store = MilvusVectorStore(host=host, port=port, collection_name=collection_name)
    vector_store.connect()
    vector_store.create_collection(dimension=dimension, drop_existing=drop_existing)
    return vector_store.insert_records(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load chunk embeddings into Milvus.")
    parser.add_argument("--input-file", type=Path, default=Path("data/processed/chunk_embeddings.jsonl"))
    parser.add_argument("--collection-name", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--drop-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    collection_name = args.collection_name or settings.milvus_chunk_collection
    host = args.host or settings.milvus_host
    port = args.port or settings.milvus_port
    inserted_count = load_embeddings_to_milvus(
        input_file=args.input_file,
        collection_name=collection_name,
        host=host,
        port=port,
        drop_existing=args.drop_existing,
    )
    print(f"Inserted {inserted_count} embeddings into Milvus collection: {collection_name}")


if __name__ == "__main__":
    main()
