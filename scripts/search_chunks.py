import argparse

from graphrag_gnn_qa.config import get_settings
from graphrag_gnn_qa.retrieval.vector_retriever import VectorRetriever
from graphrag_gnn_qa.vectorstore.embedding import SentenceTransformerEmbeddingModel
from graphrag_gnn_qa.vectorstore.milvus_client import MilvusVectorStore


def search_chunks(query: str, top_k: int) -> list[dict]:
    settings = get_settings()
    embedding_model = SentenceTransformerEmbeddingModel(model_name=settings.embedding_model)
    vector_store = MilvusVectorStore(
        host=settings.milvus_host,
        port=settings.milvus_port,
        collection_name=settings.milvus_chunk_collection,
    )
    vector_store.connect()
    retriever = VectorRetriever(embedding_model=embedding_model, vector_store=vector_store)
    chunks = retriever.retrieve(query=query, top_k=top_k)
    return [chunk.__dict__ for chunk in chunks]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search relevant chunks from Milvus by query text.")
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    top_k = args.top_k or settings.vector_top_k
    results = search_chunks(query=args.query, top_k=top_k)

    for index, result in enumerate(results, start=1):
        print(f"[{index}] score={result['score']:.4f} chunk_id={result['chunk_id']}")
        print(f"source={result['source']}")
        print(result["content"])
        print()


if __name__ == "__main__":
    main()
