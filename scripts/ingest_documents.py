import argparse
import json
from pathlib import Path

from graphrag_gnn_qa.ingestion.document_loader import DocumentLoader
from graphrag_gnn_qa.ingestion.text_splitter import TextSplitter


def iter_document_files(input_dir: Path) -> list[Path]:
    loader = DocumentLoader()
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in loader.supported_extensions
    )


def build_document_id(file_path: Path, input_dir: Path) -> str:
    relative_path = file_path.relative_to(input_dir)
    return relative_path.with_suffix("").as_posix().replace("/", "_").replace(" ", "_")


def ingest_documents(
    input_dir: Path,
    output_file: Path,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> int:
    loader = DocumentLoader()
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    document_files = iter_document_files(input_dir)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    chunk_count = 0

    with output_file.open("w", encoding="utf-8") as writer:
        for file_path in document_files:
            document = loader.load(file_path)
            document_id = build_document_id(file_path, input_dir)
            chunks = splitter.split(document.content, document_id=document_id)

            for chunk in chunks:
                record = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": document_id,
                    "content": chunk.content,
                    "source": document.source,
                    "file_name": document.file_name,
                    "file_type": document.file_type,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                }
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_count += 1

    return chunk_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load raw documents and split them into JSONL chunks.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-file", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk_count = ingest_documents(
        input_dir=args.input_dir,
        output_file=args.output_file,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Generated {chunk_count} chunks: {args.output_file}")


if __name__ == "__main__":
    main()
