import argparse
import json
from pathlib import Path
from typing import Any

import httpx


QuestionRecord = dict[str, Any]
EvaluationRecord = dict[str, Any]


def read_question_records(input_file: Path) -> list[QuestionRecord]:
    if not input_file.exists():
        raise FileNotFoundError(f"Question file not found: {input_file}")

    records = []
    for line_number, line in enumerate(input_file.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        question = str(record.get("question", "")).strip()
        if not question:
            raise ValueError(f"Missing question in {input_file}:{line_number}")
        records.append(record)
    return records


def build_retrieval_debug_payload(
    question: str,
    vector_top_k: int | None = None,
    graph_top_k: int | None = None,
    graph_max_depth: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": question}
    if vector_top_k is not None:
        payload["vector_top_k"] = vector_top_k
    if graph_top_k is not None:
        payload["graph_top_k"] = graph_top_k
    if graph_max_depth is not None:
        payload["graph_max_depth"] = graph_max_depth
    return payload


def build_qa_payload(question: str, top_k: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"question": question}
    if top_k is not None:
        payload["top_k"] = top_k
    return payload


def request_json(client: httpx.Client, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(endpoint, json=payload)
    response.raise_for_status()
    return response.json()


def build_summary(retrieval_debug: dict[str, Any], qa: dict[str, Any]) -> dict[str, Any]:
    hybrid_results = retrieval_debug.get("hybrid_results", [])
    citations = qa.get("citations", [])
    top_hybrid = hybrid_results[0] if hybrid_results else {}
    return {
        "vector_result_count": len(retrieval_debug.get("vector_results", [])),
        "graph_result_count": len(retrieval_debug.get("graph_results", [])),
        "hybrid_result_count": len(hybrid_results),
        "citation_count": len(citations),
        "top_hybrid_evidence_id": top_hybrid.get("evidence_id"),
        "top_hybrid_fusion_score": top_hybrid.get("fusion_score"),
    }


def build_run_config(
    base_url: str,
    vector_top_k: int | None = None,
    graph_top_k: int | None = None,
    graph_max_depth: int | None = None,
    qa_top_k: int | None = None,
    fusion_weights: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "base_url": base_url,
        "vector_top_k": vector_top_k,
        "graph_top_k": graph_top_k,
        "graph_max_depth": graph_max_depth,
        "qa_top_k": qa_top_k,
        "fusion_weights": fusion_weights,
    }


def evaluate_questions(
    input_file: Path,
    output_file: Path,
    base_url: str = "http://127.0.0.1:8000",
    vector_top_k: int | None = None,
    graph_top_k: int | None = None,
    graph_max_depth: int | None = None,
    qa_top_k: int | None = None,
    timeout: float = 60.0,
    transport: httpx.BaseTransport | None = None,
) -> int:
    question_records = read_question_records(input_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(base_url=base_url, timeout=timeout, transport=transport) as client:
        with output_file.open("w", encoding="utf-8") as writer:
            for index, question_record in enumerate(question_records, start=1):
                question = str(question_record["question"]).strip()
                question_id = str(question_record.get("id") or f"q{index:04d}")

                retrieval_debug = request_json(
                    client=client,
                    endpoint="/retrieval/debug",
                    payload=build_retrieval_debug_payload(
                        question=question,
                        vector_top_k=vector_top_k,
                        graph_top_k=graph_top_k,
                        graph_max_depth=graph_max_depth,
                    ),
                )
                qa = request_json(
                    client=client,
                    endpoint="/qa/ask",
                    payload=build_qa_payload(question=question, top_k=qa_top_k),
                )

                evaluation_record: EvaluationRecord = {
                    "id": question_id,
                    "question": question,
                    "expected_answer": question_record.get("expected_answer"),
                    "expected_evidence_keywords": question_record.get("expected_evidence_keywords", []),
                    "run_config": build_run_config(
                        base_url=base_url,
                        vector_top_k=vector_top_k,
                        graph_top_k=graph_top_k,
                        graph_max_depth=graph_max_depth,
                        qa_top_k=qa_top_k,
                        fusion_weights=retrieval_debug.get("fusion_weights"),
                    ),
                    "retrieval_debug": retrieval_debug,
                    "qa": qa,
                    "summary": build_summary(retrieval_debug=retrieval_debug, qa=qa),
                }
                writer.write(json.dumps(evaluation_record, ensure_ascii=False) + "\n")

    return len(question_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval debug results and QA citations for question JSONL records.")
    parser.add_argument("--input-file", type=Path, default=Path("data/eval/questions.sample.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("data/eval/retrieval_eval_results.jsonl"))
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--vector-top-k", type=int, default=None)
    parser.add_argument("--graph-top-k", type=int, default=None)
    parser.add_argument("--graph-max-depth", type=int, default=None)
    parser.add_argument("--qa-top-k", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluated_count = evaluate_questions(
        input_file=args.input_file,
        output_file=args.output_file,
        base_url=args.base_url,
        vector_top_k=args.vector_top_k,
        graph_top_k=args.graph_top_k,
        graph_max_depth=args.graph_max_depth,
        qa_top_k=args.qa_top_k,
        timeout=args.timeout,
    )
    print(f"Evaluated {evaluated_count} questions: {args.output_file}")


if __name__ == "__main__":
    main()
