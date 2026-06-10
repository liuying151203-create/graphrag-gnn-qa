import argparse
import json
import time
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


def request_json_with_latency(
    client: httpx.Client,
    endpoint: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    start_time = time.perf_counter()
    response_json = request_json(client=client, endpoint=endpoint, payload=payload)
    latency_ms = (time.perf_counter() - start_time) * 1000
    return response_json, latency_ms


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.casefold()
    if isinstance(value, dict):
        return " ".join(normalize_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(normalize_text(item) for item in value)
    return str(value).casefold()


def normalize_keywords(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    keywords = []
    for item in value:
        keyword = str(item).strip().casefold()
        if keyword and keyword not in keywords:
            keywords.append(keyword)
    return keywords


def find_matching_keywords(value: Any, keywords: list[str]) -> list[str]:
    text = normalize_text(value)
    return [keyword for keyword in keywords if keyword in text]


def keyword_recall(matched_keywords: list[str], expected_keywords: list[str]) -> float | None:
    if not expected_keywords:
        return None
    return len(matched_keywords) / len(expected_keywords)


def build_retrieval_metrics(retrieval_debug: dict[str, Any], expected_keywords: list[str]) -> dict[str, Any]:
    hybrid_results = retrieval_debug.get("hybrid_results", [])
    if not expected_keywords:
        return {
            "expected_evidence_keyword_count": 0,
            "matched_evidence_keywords": [],
            "matched_evidence_keyword_count": 0,
            "evidence_keyword_recall": None,
            "retrieval_hit": None,
            "recall_at_k": None,
            "first_relevant_rank": None,
            "mrr": None,
            "top_hybrid_keyword_hit": None,
        }

    matched_keywords = []
    first_relevant_rank = None
    for rank, evidence in enumerate(hybrid_results, start=1):
        evidence_matches = find_matching_keywords(evidence, expected_keywords)
        for keyword in evidence_matches:
            if keyword not in matched_keywords:
                matched_keywords.append(keyword)
        if evidence_matches and first_relevant_rank is None:
            first_relevant_rank = rank

    top_hybrid = hybrid_results[0] if hybrid_results else {}
    top_hybrid_keyword_hit = bool(find_matching_keywords(top_hybrid, expected_keywords))

    return {
        "expected_evidence_keyword_count": len(expected_keywords),
        "matched_evidence_keywords": matched_keywords,
        "matched_evidence_keyword_count": len(matched_keywords),
        "evidence_keyword_recall": keyword_recall(matched_keywords, expected_keywords),
        "retrieval_hit": first_relevant_rank is not None,
        "recall_at_k": 1.0 if first_relevant_rank is not None else 0.0,
        "first_relevant_rank": first_relevant_rank,
        "mrr": 1 / first_relevant_rank if first_relevant_rank is not None else 0.0,
        "top_hybrid_keyword_hit": top_hybrid_keyword_hit,
    }


def build_citation_metrics(
    retrieval_debug: dict[str, Any],
    qa: dict[str, Any],
    expected_keywords: list[str],
) -> dict[str, Any]:
    if not expected_keywords:
        return {
            "citation_keyword_hit": None,
            "matched_citation_keywords": [],
            "matched_citation_keyword_count": 0,
        }

    evidences_by_id = {
        str(evidence.get("evidence_id")): evidence for evidence in retrieval_debug.get("hybrid_results", [])
    }
    matched_keywords = []
    for citation in qa.get("citations", []):
        citation_values = [citation]
        cited_evidence = evidences_by_id.get(str(citation.get("evidence_id")))
        if cited_evidence is not None:
            citation_values.append(cited_evidence)
        for keyword in find_matching_keywords(citation_values, expected_keywords):
            if keyword not in matched_keywords:
                matched_keywords.append(keyword)

    return {
        "citation_keyword_hit": bool(matched_keywords),
        "matched_citation_keywords": matched_keywords,
        "matched_citation_keyword_count": len(matched_keywords),
    }


def build_answer_metrics(qa: dict[str, Any], answer_keywords: list[str], answer_keyword_source: str | None) -> dict[str, Any]:
    if not answer_keywords:
        return {
            "answer_keyword_source": answer_keyword_source,
            "expected_answer_keyword_count": 0,
            "matched_answer_keywords": [],
            "matched_answer_keyword_count": 0,
            "answer_keyword_recall": None,
            "answer_keyword_hit": None,
        }

    matched_keywords = find_matching_keywords(qa.get("answer"), answer_keywords)
    return {
        "answer_keyword_source": answer_keyword_source,
        "expected_answer_keyword_count": len(answer_keywords),
        "matched_answer_keywords": matched_keywords,
        "matched_answer_keyword_count": len(matched_keywords),
        "answer_keyword_recall": keyword_recall(matched_keywords, answer_keywords),
        "answer_keyword_hit": bool(matched_keywords),
    }


def build_latency_metrics(retrieval_debug_latency_ms: float, qa_latency_ms: float) -> dict[str, float]:
    return {
        "retrieval_debug_ms": round(retrieval_debug_latency_ms, 3),
        "qa_ms": round(qa_latency_ms, 3),
        "total_ms": round(retrieval_debug_latency_ms + qa_latency_ms, 3),
    }


def build_metrics(
    question_record: QuestionRecord,
    retrieval_debug: dict[str, Any],
    qa: dict[str, Any],
    retrieval_debug_latency_ms: float,
    qa_latency_ms: float,
) -> dict[str, Any]:
    expected_evidence_keywords = normalize_keywords(question_record.get("expected_evidence_keywords", []))
    expected_answer_keywords = normalize_keywords(question_record.get("expected_answer_keywords", []))
    answer_keyword_source = "expected_answer_keywords"
    if not expected_answer_keywords:
        expected_answer_keywords = expected_evidence_keywords
        answer_keyword_source = "expected_evidence_keywords" if expected_answer_keywords else None

    return {
        "retrieval": build_retrieval_metrics(
            retrieval_debug=retrieval_debug,
            expected_keywords=expected_evidence_keywords,
        ),
        "citations": build_citation_metrics(
            retrieval_debug=retrieval_debug,
            qa=qa,
            expected_keywords=expected_evidence_keywords,
        ),
        "answer": build_answer_metrics(
            qa=qa,
            answer_keywords=expected_answer_keywords,
            answer_keyword_source=answer_keyword_source,
        ),
        "latency": build_latency_metrics(
            retrieval_debug_latency_ms=retrieval_debug_latency_ms,
            qa_latency_ms=qa_latency_ms,
        ),
    }


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


def average_metric(records: list[EvaluationRecord], path: list[str]) -> float | None:
    values = []
    for record in records:
        value: Any = record
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, bool):
            values.append(1.0 if value else 0.0)
        elif isinstance(value, int | float):
            values.append(float(value))

    if not values:
        return None
    return round(sum(values) / len(values), 4)


def build_aggregate_summary(records: list[EvaluationRecord], run_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_count": len(records),
        "run_config": run_config,
        "metrics": {
            "retrieval": {
                "avg_evidence_keyword_recall": average_metric(records, ["metrics", "retrieval", "evidence_keyword_recall"]),
                "recall_at_k": average_metric(records, ["metrics", "retrieval", "recall_at_k"]),
                "mrr": average_metric(records, ["metrics", "retrieval", "mrr"]),
                "top_hybrid_keyword_hit_rate": average_metric(records, ["metrics", "retrieval", "top_hybrid_keyword_hit"]),
            },
            "citations": {
                "citation_keyword_hit_rate": average_metric(records, ["metrics", "citations", "citation_keyword_hit"]),
            },
            "answer": {
                "avg_answer_keyword_recall": average_metric(records, ["metrics", "answer", "answer_keyword_recall"]),
                "answer_keyword_hit_rate": average_metric(records, ["metrics", "answer", "answer_keyword_hit"]),
            },
            "latency": {
                "avg_retrieval_debug_ms": average_metric(records, ["metrics", "latency", "retrieval_debug_ms"]),
                "avg_qa_ms": average_metric(records, ["metrics", "latency", "qa_ms"]),
                "avg_total_ms": average_metric(records, ["metrics", "latency", "total_ms"]),
            },
        },
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
    summary_file: Path | None = Path("data/eval/retrieval_eval_summary.json"),
    progress_every: int = 1,
) -> int:
    question_records = read_question_records(input_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    evaluation_records: list[EvaluationRecord] = []
    last_run_config: dict[str, Any] | None = None

    with httpx.Client(base_url=base_url, timeout=timeout, transport=transport, trust_env=False) as client:
        with output_file.open("w", encoding="utf-8") as writer:
            for index, question_record in enumerate(question_records, start=1):
                question = str(question_record["question"]).strip()
                question_id = str(question_record.get("id") or f"q{index:04d}")

                retrieval_debug, retrieval_debug_latency_ms = request_json_with_latency(
                    client=client,
                    endpoint="/retrieval/debug",
                    payload=build_retrieval_debug_payload(
                        question=question,
                        vector_top_k=vector_top_k,
                        graph_top_k=graph_top_k,
                        graph_max_depth=graph_max_depth,
                    ),
                )
                qa, qa_latency_ms = request_json_with_latency(
                    client=client,
                    endpoint="/qa/ask",
                    payload=build_qa_payload(question=question, top_k=qa_top_k),
                )

                run_config = build_run_config(
                    base_url=base_url,
                    vector_top_k=vector_top_k,
                    graph_top_k=graph_top_k,
                    graph_max_depth=graph_max_depth,
                    qa_top_k=qa_top_k,
                    fusion_weights=retrieval_debug.get("fusion_weights"),
                )
                evaluation_record: EvaluationRecord = {
                    "id": question_id,
                    "question": question,
                    "expected_answer": question_record.get("expected_answer"),
                    "expected_evidence_keywords": question_record.get("expected_evidence_keywords", []),
                    "expected_answer_keywords": question_record.get("expected_answer_keywords", []),
                    "run_config": run_config,
                    "retrieval_debug": retrieval_debug,
                    "qa": qa,
                    "metrics": build_metrics(
                        question_record=question_record,
                        retrieval_debug=retrieval_debug,
                        qa=qa,
                        retrieval_debug_latency_ms=retrieval_debug_latency_ms,
                        qa_latency_ms=qa_latency_ms,
                    ),
                    "summary": build_summary(retrieval_debug=retrieval_debug, qa=qa),
                }
                writer.write(json.dumps(evaluation_record, ensure_ascii=False) + "\n")
                writer.flush()
                evaluation_records.append(evaluation_record)
                last_run_config = run_config
                print_progress(
                    index=index,
                    total_count=len(question_records),
                    question_id=question_id,
                    retrieval_debug_latency_ms=retrieval_debug_latency_ms,
                    qa_latency_ms=qa_latency_ms,
                    progress_every=progress_every,
                )

    if summary_file is not None:
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        aggregate_summary = build_aggregate_summary(
            records=evaluation_records,
            run_config=last_run_config
            or build_run_config(
                base_url=base_url,
                vector_top_k=vector_top_k,
                graph_top_k=graph_top_k,
                graph_max_depth=graph_max_depth,
                qa_top_k=qa_top_k,
                fusion_weights=None,
            ),
        )
        summary_file.write_text(json.dumps(aggregate_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return len(question_records)


def print_progress(
    index: int,
    total_count: int,
    question_id: str,
    retrieval_debug_latency_ms: float,
    qa_latency_ms: float,
    progress_every: int,
) -> None:
    if progress_every <= 0:
        return
    if index != total_count and index % progress_every != 0:
        return
    total_latency_ms = retrieval_debug_latency_ms + qa_latency_ms
    print(
        f"[evaluate_retrieval] {index}/{total_count} "
        f"id={question_id} "
        f"retrieval={retrieval_debug_latency_ms:.1f}ms "
        f"qa={qa_latency_ms:.1f}ms "
        f"total={total_latency_ms:.1f}ms",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval debug results and QA citations for question JSONL records.")
    parser.add_argument("--input-file", type=Path, default=Path("data/eval/questions.sample.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("data/eval/retrieval_eval_results.jsonl"))
    parser.add_argument("--summary-file", type=Path, default=Path("data/eval/retrieval_eval_summary.json"))
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--vector-top-k", type=int, default=None)
    parser.add_argument("--graph-top-k", type=int, default=None)
    parser.add_argument("--graph-max-depth", type=int, default=None)
    parser.add_argument("--qa-top-k", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress every N questions. Use 0 to disable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluated_count = evaluate_questions(
        input_file=args.input_file,
        output_file=args.output_file,
        summary_file=args.summary_file,
        base_url=args.base_url,
        vector_top_k=args.vector_top_k,
        graph_top_k=args.graph_top_k,
        graph_max_depth=args.graph_max_depth,
        qa_top_k=args.qa_top_k,
        timeout=args.timeout,
        progress_every=args.progress_every,
    )
    print(f"Evaluated {evaluated_count} questions: {args.output_file}")
    if args.summary_file is not None:
        print(f"Wrote summary: {args.summary_file}")


if __name__ == "__main__":
    main()
