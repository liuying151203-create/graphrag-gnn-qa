import json
from pathlib import Path

import httpx
import pytest

from scripts.evaluate_retrieval import (
    average_metric,
    build_aggregate_summary,
    build_metrics,
    build_qa_payload,
    build_retrieval_debug_payload,
    build_run_config,
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


def test_average_metric_ignores_missing_and_none_values() -> None:
    records = [
        {"metrics": {"retrieval": {"mrr": 1.0, "retrieval_hit": True}}},
        {"metrics": {"retrieval": {"mrr": None, "retrieval_hit": False}}},
        {"metrics": {"retrieval": {}}},
    ]

    assert average_metric(records, ["metrics", "retrieval", "mrr"]) == 1.0
    assert average_metric(records, ["metrics", "retrieval", "retrieval_hit"]) == 0.5
    assert average_metric(records, ["metrics", "answer", "answer_keyword_recall"]) is None


def test_build_aggregate_summary_averages_key_metrics() -> None:
    records = [
        {
            "metrics": {
                "retrieval": {
                    "evidence_keyword_recall": 1.0,
                    "recall_at_k": 1.0,
                    "mrr": 1.0,
                    "top_hybrid_keyword_hit": True,
                },
                "vector_retrieval": {
                    "evidence_keyword_recall": 0.5,
                    "recall_at_k": 1.0,
                    "mrr": 0.5,
                    "top_vector_keyword_hit": False,
                },
                "citations": {"citation_keyword_hit": True},
                "answer": {"answer_keyword_recall": 0.5, "answer_keyword_hit": True},
                "latency": {"retrieval_debug_ms": 10.0, "qa_ms": 20.0, "total_ms": 30.0},
            }
        },
        {
            "metrics": {
                "retrieval": {
                    "evidence_keyword_recall": 0.5,
                    "recall_at_k": 0.0,
                    "mrr": 0.0,
                    "top_hybrid_keyword_hit": False,
                },
                "vector_retrieval": {
                    "evidence_keyword_recall": 0.0,
                    "recall_at_k": 0.0,
                    "mrr": 0.0,
                    "top_vector_keyword_hit": False,
                },
                "citations": {"citation_keyword_hit": False},
                "answer": {"answer_keyword_recall": 1.0, "answer_keyword_hit": True},
                "latency": {"retrieval_debug_ms": 30.0, "qa_ms": 40.0, "total_ms": 70.0},
            }
        },
    ]

    summary = build_aggregate_summary(records, run_config={"base_url": "http://testserver"})

    assert summary == {
        "question_count": 2,
        "run_config": {"base_url": "http://testserver"},
        "metrics": {
            "retrieval": {
                "avg_evidence_keyword_recall": 0.75,
                "recall_at_k": 0.5,
                "mrr": 0.5,
                "top_hybrid_keyword_hit_rate": 0.5,
            },
            "vector_retrieval": {
                "avg_evidence_keyword_recall": 0.25,
                "recall_at_k": 0.5,
                "mrr": 0.25,
                "top_vector_keyword_hit_rate": 0.0,
            },
            "citations": {"citation_keyword_hit_rate": 0.5},
            "answer": {
                "avg_answer_keyword_recall": 0.75,
                "answer_keyword_hit_rate": 1.0,
            },
            "latency": {
                "avg_retrieval_debug_ms": 20.0,
                "avg_qa_ms": 30.0,
                "avg_total_ms": 50.0,
            },
        },
    }


def test_build_run_config_records_retrieval_parameters_and_fusion_weights() -> None:
    run_config = build_run_config(
        base_url="http://testserver",
        vector_top_k=3,
        graph_top_k=5,
        graph_max_depth=2,
        qa_top_k=3,
        fusion_weights={"score_weight": 0.7, "rank_weight": 0.3},
    )

    assert run_config == {
        "base_url": "http://testserver",
        "vector_top_k": 3,
        "graph_top_k": 5,
        "graph_max_depth": 2,
        "qa_top_k": 3,
        "fusion_weights": {"score_weight": 0.7, "rank_weight": 0.3},
    }


def test_build_metrics_records_keyword_hits_ranks_and_latency() -> None:
    metrics = build_metrics(
        question_record={
            "expected_evidence_keywords": ["GraphRAG", "retrieval"],
        },
        retrieval_debug={
            "vector_results": [
                {
                    "chunk_id": "sample_chunk_0000",
                    "content": "GraphRAG appears in vector evidence.",
                    "score": 0.9,
                }
            ],
            "hybrid_results": [
                {
                    "evidence_id": "V1",
                    "content": "Unrelated context.",
                    "fusion_score": 0.9,
                },
                {
                    "evidence_id": "V2",
                    "content": "GraphRAG combines graph retrieval and generation.",
                    "fusion_score": 0.8,
                },
            ],
        },
        qa={
            "answer": "GraphRAG improves retrieval quality.",
            "citations": [{"evidence_id": "V2"}],
        },
        retrieval_debug_latency_ms=12.3456,
        qa_latency_ms=20.1111,
    )

    assert metrics == {
        "retrieval": {
            "expected_evidence_keyword_count": 2,
            "matched_evidence_keywords": ["graphrag", "retrieval"],
            "matched_evidence_keyword_count": 2,
            "evidence_keyword_recall": 1.0,
            "retrieval_hit": True,
            "recall_at_k": 1.0,
            "first_relevant_rank": 2,
            "mrr": 0.5,
            "top_hybrid_keyword_hit": False,
        },
        "vector_retrieval": {
            "expected_evidence_keyword_count": 2,
            "matched_evidence_keywords": ["graphrag"],
            "matched_evidence_keyword_count": 1,
            "evidence_keyword_recall": 0.5,
            "retrieval_hit": True,
            "recall_at_k": 1.0,
            "first_relevant_rank": 1,
            "mrr": 1.0,
            "top_vector_keyword_hit": True,
        },
        "citations": {
            "citation_keyword_hit": True,
            "matched_citation_keywords": ["graphrag", "retrieval"],
            "matched_citation_keyword_count": 2,
        },
        "answer": {
            "answer_keyword_source": "expected_evidence_keywords",
            "expected_answer_keyword_count": 2,
            "matched_answer_keywords": ["graphrag", "retrieval"],
            "matched_answer_keyword_count": 2,
            "answer_keyword_recall": 1.0,
            "answer_keyword_hit": True,
        },
        "latency": {
            "retrieval_debug_ms": 12.346,
            "qa_ms": 20.111,
            "total_ms": 32.457,
        },
    }


def test_evaluate_questions_writes_jsonl_output(tmp_path: Path) -> None:
    input_file = tmp_path / "questions.jsonl"
    output_file = tmp_path / "results.jsonl"
    summary_file = tmp_path / "summary.json"
    input_file.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "What is GraphRAG?",
                "expected_answer": "GraphRAG combines retrieval and generation.",
                "expected_evidence_keywords": ["GraphRAG"],
                "expected_answer_keywords": ["generation"],
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
                    "fusion_weights": {"score_weight": 0.7, "rank_weight": 0.3},
                    "vector_results": [
                        {
                            "chunk_id": "sample_chunk_0000",
                            "content": "GraphRAG vector evidence.",
                        }
                    ],
                    "graph_results": [],
                    "hybrid_results": [
                        {
                            "evidence_id": "V1",
                            "fusion_score": 0.944,
                            "content": "GraphRAG retrieval evidence.",
                        }
                    ],
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
        summary_file=summary_file,
        base_url="http://testserver",
        vector_top_k=3,
        qa_top_k=3,
        transport=httpx.MockTransport(handler),
    )

    records = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert evaluated_count == 1
    assert records[0]["id"] == "q1"
    assert records[0]["question"] == "What is GraphRAG?"
    assert records[0]["expected_answer"] == "GraphRAG combines retrieval and generation."
    assert records[0]["expected_evidence_keywords"] == ["GraphRAG"]
    assert records[0]["expected_answer_keywords"] == ["generation"]
    assert records[0]["run_config"] == {
        "base_url": "http://testserver",
        "vector_top_k": 3,
        "graph_top_k": None,
        "graph_max_depth": None,
        "qa_top_k": 3,
        "fusion_weights": {"score_weight": 0.7, "rank_weight": 0.3},
    }
    assert records[0]["summary"] == {
        "vector_result_count": 1,
        "graph_result_count": 0,
        "hybrid_result_count": 1,
        "citation_count": 1,
        "top_hybrid_evidence_id": "V1",
        "top_hybrid_fusion_score": 0.944,
    }
    assert records[0]["metrics"]["retrieval"] == {
        "expected_evidence_keyword_count": 1,
        "matched_evidence_keywords": ["graphrag"],
        "matched_evidence_keyword_count": 1,
        "evidence_keyword_recall": 1.0,
        "retrieval_hit": True,
        "recall_at_k": 1.0,
        "first_relevant_rank": 1,
        "mrr": 1.0,
        "top_hybrid_keyword_hit": True,
    }
    assert records[0]["metrics"]["vector_retrieval"] == {
        "expected_evidence_keyword_count": 1,
        "matched_evidence_keywords": ["graphrag"],
        "matched_evidence_keyword_count": 1,
        "evidence_keyword_recall": 1.0,
        "retrieval_hit": True,
        "recall_at_k": 1.0,
        "first_relevant_rank": 1,
        "mrr": 1.0,
        "top_vector_keyword_hit": True,
    }
    assert records[0]["metrics"]["citations"] == {
        "citation_keyword_hit": True,
        "matched_citation_keywords": ["graphrag"],
        "matched_citation_keyword_count": 1,
    }
    assert records[0]["metrics"]["answer"] == {
        "answer_keyword_source": "expected_answer_keywords",
        "expected_answer_keyword_count": 1,
        "matched_answer_keywords": ["generation"],
        "matched_answer_keyword_count": 1,
        "answer_keyword_recall": 1.0,
        "answer_keyword_hit": True,
    }
    assert set(records[0]["metrics"]["latency"]) == {"retrieval_debug_ms", "qa_ms", "total_ms"}
    assert records[0]["metrics"]["latency"]["retrieval_debug_ms"] >= 0
    assert records[0]["metrics"]["latency"]["qa_ms"] >= 0
    assert records[0]["metrics"]["latency"]["total_ms"] >= 0
    assert summary["question_count"] == 1
    assert summary["run_config"] == records[0]["run_config"]
    assert summary["metrics"]["retrieval"] == {
        "avg_evidence_keyword_recall": 1.0,
        "recall_at_k": 1.0,
        "mrr": 1.0,
        "top_hybrid_keyword_hit_rate": 1.0,
    }
    assert summary["metrics"]["vector_retrieval"] == {
        "avg_evidence_keyword_recall": 1.0,
        "recall_at_k": 1.0,
        "mrr": 1.0,
        "top_vector_keyword_hit_rate": 1.0,
    }
    assert summary["metrics"]["citations"] == {"citation_keyword_hit_rate": 1.0}
    assert summary["metrics"]["answer"] == {
        "avg_answer_keyword_recall": 1.0,
        "answer_keyword_hit_rate": 1.0,
    }
    assert set(summary["metrics"]["latency"]) == {"avg_retrieval_debug_ms", "avg_qa_ms", "avg_total_ms"}


def test_parse_args_for_evaluate_retrieval(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_retrieval.py",
            "--input-file",
            "data/eval/questions.jsonl",
            "--output-file",
            "data/eval/results.jsonl",
            "--summary-file",
            "data/eval/summary.json",
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
    assert args.summary_file == Path("data/eval/summary.json")
    assert args.base_url == "http://localhost:8000"
    assert args.vector_top_k == 3
    assert args.graph_top_k == 5
    assert args.graph_max_depth == 2
    assert args.qa_top_k == 3
    assert args.timeout == 10
