import argparse
import json
import time
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
    progress_every: int = 1,
    resume: bool = False,
    max_retries: int = 0,
    retry_delay: float = 1.0,
) -> dict[str, int]:
    chunks = read_chunk_records(input_file)
    if limit is not None:
        chunks = chunks[:limit]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if error_file is not None:
        error_file.parent.mkdir(parents=True, exist_ok=True)
    existing_successful_chunk_ids = read_successful_chunk_ids(output_file) if resume else set()
    total_extracted_count = len(existing_successful_chunk_ids)
    new_extracted_count = 0
    error_count = 0
    total_count = len(chunks)

    error_writer = error_file.open("w", encoding="utf-8") if error_file is not None else None
    output_mode = "a" if resume else "w"
    with output_file.open(output_mode, encoding="utf-8") as writer:
        for index, chunk in enumerate(chunks, start=1):
            if chunk.get("chunk_id") in existing_successful_chunk_ids:
                print_progress(
                    index=index,
                    total_count=total_count,
                    existing_count=len(existing_successful_chunk_ids),
                    new_extracted_count=new_extracted_count,
                    error_count=error_count,
                    chunk=chunk,
                    progress_every=progress_every,
                    status="skip",
                )
                continue

            try:
                result = extract_chunk_with_retries(
                    extractor=extractor,
                    chunk=chunk,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                )
            except Exception as exc:
                if fail_fast:
                    raise
                error_count += 1
                if error_writer is not None:
                    error_writer.write(json.dumps(build_error_record(chunk, exc), ensure_ascii=False) + "\n")
                    error_writer.flush()
                print_progress(
                    index=index,
                    total_count=total_count,
                    existing_count=len(existing_successful_chunk_ids),
                    new_extracted_count=new_extracted_count,
                    error_count=error_count,
                    chunk=chunk,
                    progress_every=progress_every,
                    status="error",
                )
                continue

            writer.write(json.dumps(graph_result_to_dict(result), ensure_ascii=False) + "\n")
            writer.flush()
            new_extracted_count += 1
            total_extracted_count += 1
            print_progress(
                index=index,
                total_count=total_count,
                existing_count=len(existing_successful_chunk_ids),
                new_extracted_count=new_extracted_count,
                error_count=error_count,
                chunk=chunk,
                progress_every=progress_every,
                status="ok",
            )

    if error_writer is not None:
        error_writer.close()

    return {
        "total_chunks": total_count,
        "existing_successful_chunks": len(existing_successful_chunk_ids),
        "new_extracted_chunks": new_extracted_count,
        "total_successful_chunks": total_extracted_count,
        "failed_chunks": error_count,
    }


def read_successful_chunk_ids(output_file: Path) -> set[str]:
    if not output_file.exists():
        return set()

    chunk_ids = set()
    for line in output_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        chunk_id = str(record.get("chunk_id", "")).strip()
        if chunk_id:
            chunk_ids.add(chunk_id)
    return chunk_ids


def extract_chunk_with_retries(
    extractor: GraphExtractor,
    chunk: dict,
    max_retries: int = 0,
    retry_delay: float = 1.0,
) -> object:
    attempt = 0
    while True:
        try:
            return extractor.extract_from_chunk(chunk)
        except Exception:
            if attempt >= max_retries:
                raise
            attempt += 1
            if retry_delay > 0:
                time.sleep(retry_delay)


def print_progress(
    index: int,
    total_count: int,
    existing_count: int,
    new_extracted_count: int,
    error_count: int,
    chunk: dict,
    progress_every: int,
    status: str = "ok",
) -> None:
    if progress_every <= 0:
        return
    if index != total_count and index % progress_every != 0:
        return
    print(
        f"[extract_graph] {index}/{total_count} "
        f"status={status} existing={existing_count} new_ok={new_extracted_count} errors={error_count} "
        f"chunk_id={chunk.get('chunk_id')}",
        flush=True,
    )


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
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress every N chunks. Use 0 to disable.")
    parser.add_argument("--resume", action="store_true", help="Append new records and skip chunks already in output-file.")
    parser.add_argument("--max-retries", type=int, default=0, help="Retry each failed chunk up to N times.")
    parser.add_argument("--retry-delay", type=float, default=1.0, help="Seconds to wait between retry attempts.")
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
    summary = extract_graph(
        input_file=args.input_file,
        output_file=args.output_file,
        extractor=extractor,
        limit=args.limit,
        error_file=args.error_file,
        fail_fast=args.fail_fast,
        progress_every=args.progress_every,
        resume=args.resume,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )
    print(
        "Graph extraction summary: "
        f"total={summary['total_chunks']} "
        f"existing={summary['existing_successful_chunks']} "
        f"new={summary['new_extracted_chunks']} "
        f"successful={summary['total_successful_chunks']} "
        f"failed={summary['failed_chunks']} "
        f"output={args.output_file}"
    )
    if summary["failed_chunks"]:
        print(f"Skipped {summary['failed_chunks']} chunks with extraction errors: {args.error_file}")


if __name__ == "__main__":
    main()
