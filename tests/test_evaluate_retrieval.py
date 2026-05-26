import json
from pathlib import Path

import httpx
import pytest

from scripts.evaluate_retrieval import (
    build_qa_payload,
    build_retrieval_debug_payload,
    build_summary,
    evaluate_questions,
    parse_args,
    read_question_records,
)


def test_read_question_records_reads_jsonl(tmp_path: Path) -> None:
    input_file = tmp_path / "questions.jsonl"
    input_file.write_text(
        json.dumps({"id": "q1", "question": "What is GraphRAG?"}) + "\n",
        encoding="utf-8",
    )

    records = read_question_records(input_file)

    assert records == [{"id": "q1", "question": "What is GraphRAG?"}]


def test_read_question_records_rejects_missing_question(tmp_path: Path) -> None:
    input_file = tmp_path / "questions.jsonl"
    input_file.write_text(json.dumps({"id": "q1"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing question"):
        read_question_records(input_file)


def test_build_retrieval_debug_payload_omits_none_values() -> None:
    payload = build_retrieval_debug_payload(
        question="What is GraphRAG?",
        vector_top_k=3,
        graph_top_k=None,
        graph_max_depth=2,
    )

    assert payload == {
        "query": "What is GraphRAG?",
        "vector_top_k": 3,
        "graph_max_depth": 2,
    }


def test_build_qa_payload_omits_none_top_k() -> None:
    assert build_qa_payload(question="What is GraphRAG?") == {"question": "What is GraphRAG?"}


def test_build_summary_counts_results_and_top_hybrid() -> None:
    summary = build_summary(
        retrieval_debug={
            "vector_results": [{"chunk_id": "c1"}],
            "graph_results": [{"relation_type": "USES"}],
            "hybrid_results": [{"evidence_id": "V1+G1", "fusion_score": 0.9}],
        },
        qa={"citations": [{"evidence_id": "V1+G1"}]},
    )

    assert summary == {
        "vector_result_count": 1,
        "graph_result_count": 1,
        "hybrid_result_count": 1,
        "citation_count": 1,
        "top_hybrid_evidence_id": "V1+G1",
        "top_hybrid_fusion_score": 0.9,
    }


def test_evaluate_questions_writes_jsonl_output(tmp_path: Path) -> None:
    input_file = tmp_path / "questions.jsonl"
    output_file = tmp_path / "results.jsonl"
    input_file.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "What is GraphRAG?",
                "expected_answer": "GraphRAG combines retrieval and generation.",
                "expected_evidence_keywords": ["GraphRAG"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/retrieval/debug":
            assert json.loads(request.content) == {"query": "What is GraphRAG?", "vector_top_k": 3}
            return httpx.Response(
                200,
                json={
                    "query": "What is GraphRAG?",
                    "vector_results": [{"chunk_id": "sample_chunk_0000"}],
                    "graph_results": [],
                    "hybrid_results": [{"evidence_id": "V1", "fusion_score": 0.944}],
                },
            )
        if request.url.path == "/qa/ask":
            assert json.loads(request.content) == {"question": "What is GraphRAG?", "top_k": 3}
            return httpx.Response(
                200,
                json={
                    "question": "What is GraphRAG?",
                    "answer": "GraphRAG combines retrieval and generation.",
                    "sources": [],
                    "graph_sources": [],
                    "citations": [{"evidence_id": "V1"}],
                },
            )
        return httpx.Response(404)

    evaluated_count = evaluate_questions(
        input_file=input_file,
        output_file=output_file,
        base_url="http://testserver",
        vector_top_k=3,
        qa_top_k=3,
        transport=httpx.MockTransport(handler),
    )

    records = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
    assert evaluated_count == 1
    assert records[0]["id"] == "q1"
    assert records[0]["question"] == "What is GraphRAG?"
    assert records[0]["expected_answer"] == "GraphRAG combines retrieval and generation."
    assert records[0]["expected_evidence_keywords"] == ["GraphRAG"]
    assert records[0]["summary"] == {
        "vector_result_count": 1,
        "graph_result_count": 0,
        "hybrid_result_count": 1,
        "citation_count": 1,
        "top_hybrid_evidence_id": "V1",
        "top_hybrid_fusion_score": 0.944,
    }


def test_parse_args_for_evaluate_retrieval(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_retrieval.py",
            "--input-file",
            "data/eval/questions.jsonl",
            "--output-file",
            "data/eval/results.jsonl",
            "--base-url",
            "http://localhost:8000",
            "--vector-top-k",
            "3",
            "--graph-top-k",
            "5",
            "--graph-max-depth",
            "2",
            "--qa-top-k",
            "3",
            "--timeout",
            "10",
        ],
    )

    args = parse_args()

    assert args.input_file == Path("data/eval/questions.jsonl")
    assert args.output_file == Path("data/eval/results.jsonl")
    assert args.base_url == "http://localhost:8000"
    assert args.vector_top_k == 3
    assert args.graph_top_k == 5
    assert args.graph_max_depth == 2
    assert args.qa_top_k == 3
    assert args.timeout == 10
