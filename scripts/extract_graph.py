import argparse
import json
from pathlib import Path

from graphrag_gnn_qa.config import get_settings
from graphrag_gnn_qa.graph.extractor import GraphExtractor, graph_result_to_dict
from graphrag_gnn_qa.llm.client import OpenAICompatibleLLMClient


def read_chunk_records(input_file: Path) -> list[dict]:
    if not input_file.exists():
        raise FileNotFoundError(f"Chunk file not found: {input_file}")

    return [
        json.loads(line)
        for line in input_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def extract_graph(
    input_file: Path,
    output_file: Path,
    extractor: GraphExtractor,
    limit: int | None = None,
    error_file: Path | None = None,
    fail_fast: bool = False,
) -> int:
    chunks = read_chunk_records(input_file)
    if limit is not None:
        chunks = chunks[:limit]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if error_file is not None:
        error_file.parent.mkdir(parents=True, exist_ok=True)
    extracted_count = 0

    error_writer = error_file.open("w", encoding="utf-8") if error_file is not None else None
    with output_file.open("w", encoding="utf-8") as writer:
        for chunk in chunks:
            try:
                result = extractor.extract_from_chunk(chunk)
            except Exception as exc:
                if fail_fast:
                    raise
                if error_writer is not None:
                    error_writer.write(json.dumps(build_error_record(chunk, exc), ensure_ascii=False) + "\n")
                continue

            writer.write(json.dumps(graph_result_to_dict(result), ensure_ascii=False) + "\n")
            extracted_count += 1

    if error_writer is not None:
        error_writer.close()

    return extracted_count


def build_error_record(chunk: dict, exc: Exception) -> dict:
    return {
        "chunk_id": chunk.get("chunk_id"),
        "document_id": chunk.get("document_id"),
        "source": chunk.get("source"),
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "content_preview": str(chunk.get("content", ""))[:300],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract graph entities and relations from chunk JSONL records.")
    parser.add_argument("--input-file", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("data/processed/graph_triples.jsonl"))
    parser.add_argument("--error-file", type=Path, default=Path("data/processed/graph_extraction_errors.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    llm_client = OpenAICompatibleLLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )
    extractor = GraphExtractor(llm_client=llm_client)
    extracted_count = extract_graph(
        input_file=args.input_file,
        output_file=args.output_file,
        extractor=extractor,
        limit=args.limit,
        error_file=args.error_file,
        fail_fast=args.fail_fast,
    )
    print(f"Extracted graph records from {extracted_count} chunks: {args.output_file}")
    if args.error_file.exists():
        error_count = len([line for line in args.error_file.read_text(encoding="utf-8").splitlines() if line.strip()])
        if error_count:
            print(f"Skipped {error_count} chunks with extraction errors: {args.error_file}")


if __name__ == "__main__":
    main()
