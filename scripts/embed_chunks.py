import argparse
import json
from pathlib import Path

from graphrag_gnn_qa.config import get_settings
from graphrag_gnn_qa.vectorstore.embedding import EmbeddingModel, SentenceTransformerEmbeddingModel


def read_jsonl(file_path: Path) -> list[dict]:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    records = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def batched(records: list[dict], batch_size: int) -> list[list[dict]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]


def embed_chunks(
    input_file: Path,
    output_file: Path,
    embedding_model: EmbeddingModel,
    batch_size: int = 16,
) -> int:
    records = read_jsonl(input_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    embedded_count = 0

    with output_file.open("w", encoding="utf-8") as writer:
        for batch in batched(records, batch_size):
            texts = [record["content"] for record in batch]
            embeddings = embedding_model.embed_texts(texts)

            for record, embedding in zip(batch, embeddings):
                output_record = {
                    "chunk_id": record["chunk_id"],
                    "document_id": record["document_id"],
                    "content": record["content"],
                    "source": record["source"],
                    "file_name": record["file_name"],
                    "file_type": record["file_type"],
                    "embedding": embedding,
                }
                writer.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                embedded_count += 1

    return embedded_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embeddings for chunk JSONL records.")
    parser.add_argument("--input-file", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("data/processed/chunk_embeddings.jsonl"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    model_name = args.model_name or settings.embedding_model
    embedding_model = SentenceTransformerEmbeddingModel(model_name=model_name)
    embedded_count = embed_chunks(
        input_file=args.input_file,
        output_file=args.output_file,
        embedding_model=embedding_model,
        batch_size=args.batch_size,
    )
    print(f"Generated {embedded_count} embeddings: {args.output_file}")


if __name__ == "__main__":
    main()
