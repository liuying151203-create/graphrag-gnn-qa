import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PREFERRED_PRESET_IDS = (
    "domain_q001",
    "domain_q006",
    "domain_q010",
    "domain_q014",
    "domain_q020",
    "domain_q024",
    "domain_q026",
)
DOCUMENT_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}
NODE_COLORS = {
    "Method": "#CDEBE5",
    "Task": "#FCE7B2",
    "Dataset": "#D9E8F7",
    "Metric": "#F7D6D0",
    "Concept": "#E5E7EB",
    "Paper": "#DDE4F4",
}


@dataclass(frozen=True)
class PresetQuestion:
    question_id: str
    question: str
    topic: str
    category: str
    expected_answer_keywords: tuple[str, ...]


@dataclass(frozen=True)
class DatasetStats:
    dataset_name: str
    document_count: int
    chunk_count: int
    entity_count: int
    relation_count: int
    question_count: int


def load_preset_questions(file_path: Path, limit: int = 7) -> list[PresetQuestion]:
    if limit <= 0:
        raise ValueError("limit must be greater than 0")
    if not file_path.exists():
        return []

    questions = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        question = str(record.get("question", "")).strip()
        if not question:
            continue
        metadata = record.get("metadata") or {}
        questions.append(
            PresetQuestion(
                question_id=str(record.get("id", "")).strip() or f"question_{len(questions) + 1}",
                question=question,
                topic=str(metadata.get("topic", "General")),
                category=str(metadata.get("category", "unknown")),
                expected_answer_keywords=tuple(str(item) for item in record.get("expected_answer_keywords", [])),
            )
        )

    by_id = {question.question_id: question for question in questions}
    preferred = [by_id[question_id] for question_id in PREFERRED_PRESET_IDS if question_id in by_id]
    if preferred:
        return preferred[:limit]
    return questions[:limit]


def load_demo_snapshot(file_path: Path) -> dict[str, Any] | None:
    if not file_path.exists():
        return None
    try:
        snapshot = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(snapshot, dict) or not str(snapshot.get("question") or "").strip():
        return None
    return snapshot


def collect_dataset_stats(project_root: Path, question_file: Path) -> DatasetStats:
    raw_dir = project_root / "data" / "raw"
    documents = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() in DOCUMENT_SUFFIXES
    ] if raw_dir.exists() else []
    entity_count, relation_count = _count_graph_items(
        project_root / "data" / "processed" / "graph_triples.jsonl"
    )
    dataset_name = question_file.stem.removeprefix("questions.")
    if len(documents) == 1 and documents[0].stem.casefold() == "sample":
        dataset_name = "sample"
    return DatasetStats(
        dataset_name=dataset_name,
        document_count=len(documents),
        chunk_count=_count_jsonl_records(project_root / "data" / "processed" / "chunks.jsonl"),
        entity_count=entity_count,
        relation_count=relation_count,
        question_count=_count_jsonl_records(question_file),
    )


def build_comparison_rows(
    retrieval_debug: dict[str, Any],
    qa: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    vector_results = retrieval_debug.get("vector_results") or []
    hybrid_results = retrieval_debug.get("hybrid_results") or []
    timings = retrieval_debug.get("timings") or {}
    citations = (qa or {}).get("citations") or []
    return [
        {
            "方法": "Vector-only",
            "证据数量": len(vector_results),
            "Top 证据": _top_evidence_label(vector_results, "score"),
            "Top 分数": _top_score(vector_results, "score"),
            "引用数量": 0,
            "检索耗时 (ms)": _number_or_none(timings.get("vector_ms")),
        },
        {
            "方法": "GraphRAG hybrid",
            "证据数量": len(hybrid_results),
            "Top 证据": _top_evidence_label(hybrid_results, "fusion_score"),
            "Top 分数": _top_score(hybrid_results, "fusion_score"),
            "引用数量": len(citations),
            "检索耗时 (ms)": _number_or_none(timings.get("total_ms")),
        },
    ]


def build_timing_rows(qa: dict[str, Any] | None) -> list[dict[str, Any]]:
    timings = (qa or {}).get("timings") or {}
    labels = (
        ("vector_ms", "Vector"),
        ("graph_ms", "Graph"),
        ("fusion_ms", "Fusion"),
        ("rerank_ms", "Rerank"),
        ("llm_ms", "LLM"),
    )
    return [
        {"阶段": label, "耗时 (ms)": float(timings.get(key, 0.0) or 0.0)}
        for key, label in labels
    ]


def find_citation_evidence(
    citation: dict[str, Any],
    retrieval_debug: dict[str, Any],
) -> dict[str, Any] | None:
    evidence_id = citation.get("evidence_id")
    document_id = citation.get("document_id")
    chunk_id = citation.get("chunk_id")
    for evidence in retrieval_debug.get("hybrid_results") or []:
        if evidence.get("evidence_id") == evidence_id:
            return evidence
    for evidence in retrieval_debug.get("hybrid_results") or []:
        if evidence.get("document_id") == document_id and evidence.get("chunk_id") == chunk_id:
            return evidence
    return None


def build_graph_dot(graph_results: list[dict[str, Any]], max_relations: int = 30) -> str:
    if max_relations <= 0:
        raise ValueError("max_relations must be greater than 0")
    nodes: dict[str, tuple[str, str]] = {}
    edges = []
    seen_edges = set()
    for relation in graph_results[:max_relations]:
        source_id = _node_id(relation, "source")
        target_id = _node_id(relation, "target")
        source_name = str(relation.get("source_name") or source_id)
        target_name = str(relation.get("target_name") or target_id)
        source_type = str(relation.get("source_type") or "Concept")
        target_type = str(relation.get("target_type") or "Concept")
        relation_type = str(relation.get("relation_type") or "RELATED_TO")
        nodes[source_id] = (source_name, source_type)
        nodes[target_id] = (target_name, target_type)
        edge_key = (source_id, relation_type, target_id)
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            edges.append(edge_key)

    lines = [
        "digraph G {",
        '  graph [bgcolor="transparent", rankdir="LR", pad="0.25", nodesep="0.45", ranksep="0.7"];',
        '  node [shape="box", style="rounded,filled", color="#94A3B8", fontname="Arial", fontsize="10"];',
        '  edge [color="#64748B", fontcolor="#475569", fontname="Arial", fontsize="9", arrowsize="0.7"];',
    ]
    for node_id, (name, node_type) in nodes.items():
        color = NODE_COLORS.get(node_type, "#E5E7EB")
        label = f"{name} [{node_type}]"
        lines.append(
            f'  "{_dot_escape(node_id)}" [label="{_dot_escape(label)}", fillcolor="{color}"];'
        )
    for source_id, relation_type, target_id in edges:
        lines.append(
            f'  "{_dot_escape(source_id)}" -> "{_dot_escape(target_id)}" '
            f'[label="{_dot_escape(relation_type)}"];'
        )
    lines.append("}")
    return "\n".join(lines)


def _count_jsonl_records(file_path: Path) -> int:
    if not file_path.exists():
        return 0
    return sum(1 for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip())


def _count_graph_items(file_path: Path) -> tuple[int, int]:
    if not file_path.exists():
        return 0, 0
    entity_ids = set()
    relation_count = 0
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        for entity in record.get("entities") or []:
            entity_type = str(entity.get("type") or "Concept")
            entity_name = " ".join(str(entity.get("name") or "").casefold().split())
            if entity_name:
                entity_ids.add((entity_type, entity_name))
        relation_count += len(record.get("relations") or [])
    return len(entity_ids), relation_count


def _node_id(relation: dict[str, Any], prefix: str) -> str:
    node_id = str(relation.get(f"{prefix}_id") or "").strip()
    if node_id:
        return node_id
    node_type = str(relation.get(f"{prefix}_type") or "Concept")
    node_name = str(relation.get(f"{prefix}_name") or prefix)
    return f"{node_type}:{node_name}"


def _dot_escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\r", " ").replace("\n", " ")


def _top_evidence_label(results: list[dict[str, Any]], score_key: str) -> str:
    if not results:
        return "-"
    top_result = max(results, key=lambda item: float(item.get(score_key, 0.0) or 0.0))
    source = str(top_result.get("source") or "unknown")
    chunk_id = str(top_result.get("chunk_id") or "-")
    return f"{Path(source).name} / {chunk_id}"


def _top_score(results: list[dict[str, Any]], score_key: str) -> float | None:
    if not results:
        return None
    return round(max(float(item.get(score_key, 0.0) or 0.0) for item in results), 4)


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, int | float):
        return round(float(value), 3)
    return None
