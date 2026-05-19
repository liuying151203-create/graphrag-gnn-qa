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


def extract_graph(input_file: Path, output_file: Path, extractor: GraphExtractor, limit: int | None = None) -> int:
    chunks = read_chunk_records(input_file)
    if limit is not None:
        chunks = chunks[:limit]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    extracted_count = 0

    with output_file.open("w", encoding="utf-8") as writer:
        for chunk in chunks:
            result = extractor.extract_from_chunk(chunk)
            writer.write(json.dumps(graph_result_to_dict(result), ensure_ascii=False) + "\n")
            extracted_count += 1

    return extracted_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract graph entities and relations from chunk JSONL records.")
    parser.add_argument("--input-file", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("data/processed/graph_triples.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
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
    )
    print(f"Extracted graph records from {extracted_count} chunks: {args.output_file}")


if __name__ == "__main__":
    main()
