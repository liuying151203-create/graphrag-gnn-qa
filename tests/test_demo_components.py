import json
from pathlib import Path

import pytest

from demo.components import (
    build_comparison_rows,
    build_graph_dot,
    build_timing_rows,
    collect_dataset_stats,
    find_citation_evidence,
    load_demo_snapshot,
    load_preset_questions,
)


def test_load_preset_questions_uses_curated_order(tmp_path: Path) -> None:
    question_file = tmp_path / "questions.domain_mini.jsonl"
    records = [
        {
            "id": "domain_q010",
            "question": "RoHe question",
            "expected_answer_keywords": ["attention"],
            "metadata": {"topic": "RoHe", "category": "fact"},
        },
        {
            "id": "domain_q001",
            "question": "HAN question",
            "expected_answer_keywords": ["heterogeneous graph"],
            "metadata": {"topic": "HAN", "category": "fact"},
        },
    ]
    question_file.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    questions = load_preset_questions(question_file)

    assert [question.question_id for question in questions] == ["domain_q001", "domain_q010"]
    assert questions[0].topic == "HAN"
    assert questions[0].expected_answer_keywords == ("heterogeneous graph",)


def test_load_preset_questions_rejects_invalid_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_preset_questions(tmp_path / "questions.jsonl", limit=0)


def test_load_demo_snapshot_validates_required_question(tmp_path: Path) -> None:
    snapshot_file = tmp_path / "snapshot.json"
    snapshot_file.write_text(json.dumps({"question": "What is GraphRAG?", "snapshot": True}), encoding="utf-8")

    assert load_demo_snapshot(snapshot_file) == {"question": "What is GraphRAG?", "snapshot": True}

    snapshot_file.write_text("{}", encoding="utf-8")
    assert load_demo_snapshot(snapshot_file) is None


def test_collect_dataset_stats_counts_local_snapshot(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    eval_dir = tmp_path / "data" / "eval"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (raw_dir / "paper.pdf").write_bytes(b"pdf")
    (raw_dir / "notes.md").write_text("notes", encoding="utf-8")
    (raw_dir / "ignored.csv").write_text("data", encoding="utf-8")
    (processed_dir / "chunks.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    graph_records = [
        {
            "entities": [
                {"type": "Method", "name": "GraphRAG"},
                {"type": "Task", "name": "Question Answering"},
            ],
            "relations": [{"relation_type": "SOLVES_TASK"}],
        },
        {
            "entities": [{"type": "Method", "name": "graphrag"}],
            "relations": [{"relation_type": "USES_METHOD"}],
        },
    ]
    (processed_dir / "graph_triples.jsonl").write_text(
        "\n".join(json.dumps(record) for record in graph_records),
        encoding="utf-8",
    )
    question_file = eval_dir / "questions.demo.jsonl"
    question_file.write_text("{}\n{}\n{}\n", encoding="utf-8")

    stats = collect_dataset_stats(tmp_path, question_file)

    assert stats.dataset_name == "demo"
    assert stats.document_count == 2
    assert stats.chunk_count == 2
    assert stats.entity_count == 2
    assert stats.relation_count == 2
    assert stats.question_count == 3


def test_build_comparison_rows_uses_vector_and_hybrid_results() -> None:
    retrieval_debug = {
        "vector_results": [
            {"score": 0.8, "source": "paper.pdf", "chunk_id": "chunk_1"},
            {"score": 0.9, "source": "paper.pdf", "chunk_id": "chunk_2"},
        ],
        "hybrid_results": [
            {"fusion_score": 0.95, "source": "paper.pdf", "chunk_id": "chunk_2"},
        ],
        "timings": {"vector_ms": 12.3456, "total_ms": 20.4567},
    }
    qa = {"citations": [{"evidence_id": "V2+G1"}]}

    rows = build_comparison_rows(retrieval_debug, qa)

    assert rows[0] == {
        "方法": "Vector-only",
        "证据数量": 2,
        "Top 证据": "paper.pdf / chunk_2",
        "Top 分数": 0.9,
        "引用数量": 0,
        "检索耗时 (ms)": 12.346,
    }
    assert rows[1]["方法"] == "GraphRAG hybrid"
    assert rows[1]["引用数量"] == 1
    assert rows[1]["检索耗时 (ms)"] == 20.457


def test_build_timing_rows_keeps_stage_order() -> None:
    rows = build_timing_rows(
        {
            "timings": {
                "vector_ms": 1,
                "graph_ms": 2,
                "fusion_ms": 3,
                "rerank_ms": 4,
                "llm_ms": 5,
            }
        }
    )

    assert [row["阶段"] for row in rows] == ["Vector", "Graph", "Fusion", "Rerank", "LLM"]
    assert [row["耗时 (ms)"] for row in rows] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_find_citation_evidence_matches_id_then_chunk() -> None:
    retrieval_debug = {
        "hybrid_results": [
            {"evidence_id": "V1", "document_id": "doc", "chunk_id": "chunk_1"},
            {"evidence_id": "V2+G1", "document_id": "doc", "chunk_id": "chunk_2"},
        ]
    }

    exact = find_citation_evidence({"evidence_id": "V2+G1"}, retrieval_debug)
    fallback = find_citation_evidence(
        {"evidence_id": "missing", "document_id": "doc", "chunk_id": "chunk_1"},
        retrieval_debug,
    )

    assert exact == retrieval_debug["hybrid_results"][1]
    assert fallback == retrieval_debug["hybrid_results"][0]


def test_build_graph_dot_deduplicates_and_escapes_graph_data() -> None:
    relation = {
        "source_id": "Method:rohe",
        "source_name": 'Ro"He',
        "source_type": "Method",
        "relation_type": "SOLVES_TASK",
        "target_id": "Task:defense",
        "target_name": "Adversarial defense",
        "target_type": "Task",
    }

    dot = build_graph_dot([relation, relation])

    assert dot.startswith("digraph G {")
    assert 'label="Ro\\"He [Method]"' in dot
    assert dot.count('label="SOLVES_TASK"') == 1
    assert 'fillcolor="#CDEBE5"' in dot


def test_build_graph_dot_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError):
        build_graph_dot([], max_relations=0)
