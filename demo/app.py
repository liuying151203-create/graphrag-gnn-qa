from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from demo.api_client import DemoApiError, GraphRAGApiClient
from demo.components import (
    PresetQuestion,
    build_comparison_rows,
    build_graph_dot,
    build_timing_rows,
    collect_dataset_stats,
    find_citation_evidence,
    load_demo_snapshot,
    load_preset_questions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTION_FILE = PROJECT_ROOT / "data" / "eval" / "questions.domain_mini.jsonl"
SNAPSHOT_FILE = PROJECT_ROOT / "data" / "demo" / "sample_snapshot.json"
DEFAULT_API_URL = os.getenv("DEMO_API_BASE_URL", "http://127.0.0.1:8000")
COMPONENT_LABELS = {
    "api": "API",
    "embedding": "Embedding",
    "milvus": "Milvus",
    "neo4j": "Neo4j",
    "reranker": "Reranker",
    "llm": "LLM",
}
STATUS_LABELS = {
    "ready": "READY",
    "unavailable": "DOWN",
    "not_configured": "NOT CONFIGURED",
    "unknown": "UNKNOWN",
}

st.set_page_config(
    page_title="GraphRAG Research Workbench",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=10, show_spinner=False)
def fetch_readiness(base_url: str) -> dict[str, Any]:
    client = GraphRAGApiClient(base_url=base_url, timeout=10)
    try:
        return client.readiness().data
    finally:
        client.close()


@st.cache_data(show_spinner=False)
def load_demo_data() -> tuple[list[PresetQuestion], Any, dict[str, Any] | None]:
    presets = load_preset_questions(QUESTION_FILE)
    stats = collect_dataset_stats(PROJECT_ROOT, QUESTION_FILE)
    snapshot = load_demo_snapshot(SNAPSHOT_FILE)
    if snapshot:
        presets.insert(
            0,
            PresetQuestion(
                question_id="sample_snapshot",
                question=str(snapshot["question"]),
                topic="GraphRAG sample",
                category="demo_snapshot",
                expected_answer_keywords=("vector search", "graph traversal"),
            ),
        )
    return presets, stats, snapshot


def run_analysis(
    base_url: str,
    question: str,
    vector_top_k: int,
    graph_top_k: int,
    graph_max_depth: int,
) -> dict[str, Any]:
    client = GraphRAGApiClient(base_url=base_url, timeout=180)
    retrieval_debug = None
    qa = None
    client_latencies = {}
    errors = []
    try:
        try:
            result = client.debug_retrieval(
                query=question,
                vector_top_k=vector_top_k,
                graph_top_k=graph_top_k,
                graph_max_depth=graph_max_depth,
            )
            retrieval_debug = result.data
            client_latencies["retrieval_debug_ms"] = result.latency_ms
        except DemoApiError as exc:
            errors.append(str(exc))

        try:
            result = client.ask(question=question, top_k=vector_top_k)
            qa = result.data
            client_latencies["qa_ms"] = result.latency_ms
        except DemoApiError as exc:
            errors.append(str(exc))
    finally:
        client.close()
    return {
        "question": question,
        "retrieval_debug": retrieval_debug,
        "qa": qa,
        "client_latencies": client_latencies,
        "errors": errors,
        "stale": False,
    }


def render_service_status(readiness: dict[str, Any] | None, error: str | None) -> None:
    components = (readiness or {}).get("components") or {}
    if not components:
        components = {
            name: {
                "status": "unavailable" if name == "api" else "unknown",
                "detail": error or "No runtime status",
            }
            for name in COMPONENT_LABELS
        }

    for row_start in range(0, len(COMPONENT_LABELS), 3):
        columns = st.columns(3)
        component_names = list(COMPONENT_LABELS)[row_start : row_start + 3]
        for column, name in zip(columns, component_names):
            component = components.get(name) or {"status": "unknown", "detail": "No status"}
            component_status = str(component.get("status") or "unknown")
            label = COMPONENT_LABELS[name]
            status_label = STATUS_LABELS.get(component_status, component_status.upper())
            detail = html.escape(str(component.get("detail") or ""))
            with column:
                st.markdown(
                    f"""
                    <div class="service-line">
                      <span class="status-dot status-{html.escape(component_status)}"></span>
                      <span class="service-name">{html.escape(label)}</span>
                      <strong>{html.escape(status_label)}</strong>
                      <span class="service-detail" title="{detail}">{detail}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_answer(result: dict[str, Any]) -> None:
    qa = result.get("qa")
    retrieval_debug = result.get("retrieval_debug") or {}
    st.subheader("答案")
    st.caption(f"结果问题：{result.get('question') or 'unknown'}")
    if result.get("stale"):
        st.warning("当前显示上一次成功结果。")
    if result.get("snapshot"):
        st.warning("当前显示已保存演示快照，不是实时请求。")
    for error in result.get("errors") or []:
        st.warning(error)

    answer_column, timing_column = st.columns([1.8, 1])
    with answer_column:
        if qa:
            st.markdown(str(qa.get("answer") or "未返回答案。"))
            citations = qa.get("citations") or []
            metrics = st.columns(3)
            metrics[0].metric("总耗时", f"{float((qa.get('timings') or {}).get('total_ms', 0.0)):.1f} ms")
            metrics[1].metric("引用", len(citations))
            metrics[2].metric("混合证据", len(retrieval_debug.get("hybrid_results") or []))
            render_citation(citations, retrieval_debug)
        else:
            st.info("问答结果不可用，检索证据仍可查看。")

    with timing_column:
        timing_rows = build_timing_rows(qa)
        timing_frame = pd.DataFrame(timing_rows).set_index("阶段")
        st.bar_chart(timing_frame, color="#0F766E", height=230)


def render_citation(citations: list[dict[str, Any]], retrieval_debug: dict[str, Any]) -> None:
    if not citations:
        return
    selected_index = st.selectbox(
        "Citation",
        options=list(range(len(citations))),
        format_func=lambda index: _citation_label(citations[index]),
        key="selected_citation",
    )
    citation = citations[selected_index]
    evidence = find_citation_evidence(citation, retrieval_debug)
    st.caption(
        f"{citation.get('source', 'unknown')} · {citation.get('chunk_id', '-')} · "
        f"fusion {float(citation.get('fusion_score', 0.0)):.4f}"
    )
    if evidence:
        st.write(str(evidence.get("content") or ""))


def render_comparison(result: dict[str, Any]) -> None:
    retrieval_debug = result.get("retrieval_debug")
    if not retrieval_debug:
        return
    st.subheader("Vector-only 与 GraphRAG")
    comparison_rows = build_comparison_rows(retrieval_debug, result.get("qa"))
    st.dataframe(
        comparison_rows,
        hide_index=True,
        width="stretch",
        column_config={
            "Top 分数": st.column_config.NumberColumn(format="%.4f"),
            "检索耗时 (ms)": st.column_config.NumberColumn(format="%.3f"),
        },
    )


def render_evidence_tabs(result: dict[str, Any]) -> None:
    retrieval_debug = result.get("retrieval_debug")
    if not retrieval_debug:
        return
    qa = result.get("qa")
    hybrid_tab, vector_tab, graph_tab, graph_view_tab, raw_tab = st.tabs(
        ["Hybrid Evidence", "Vector Results", "Graph Results", "Graph View", "Raw Response"]
    )

    with hybrid_tab:
        hybrid_results = retrieval_debug.get("hybrid_results") or []
        st.dataframe(
            [_hybrid_row(item) for item in hybrid_results],
            hide_index=True,
            width="stretch",
        )
        _render_selected_evidence(hybrid_results, "hybrid_evidence")

    with vector_tab:
        vector_results = retrieval_debug.get("vector_results") or []
        st.dataframe(
            [_vector_row(index, item) for index, item in enumerate(vector_results, start=1)],
            hide_index=True,
            width="stretch",
        )
        _render_selected_evidence(vector_results, "vector_evidence")

    with graph_tab:
        graph_results = retrieval_debug.get("graph_results") or []
        st.dataframe(
            [_graph_row(index, item) for index, item in enumerate(graph_results, start=1)],
            hide_index=True,
            width="stretch",
        )
        _render_selected_evidence(graph_results, "graph_evidence", content_key="evidence")

    with graph_view_tab:
        graph_results = retrieval_debug.get("graph_results") or []
        if graph_results:
            st.graphviz_chart(build_graph_dot(graph_results), width="stretch")
        else:
            st.info("当前问题没有命中图谱关系。")

    with raw_tab:
        st.json(
            {
                "retrieval_debug": retrieval_debug,
                "qa": qa,
                "client_latencies": result.get("client_latencies") or {},
            },
            expanded=False,
        )


def _render_selected_evidence(
    evidences: list[dict[str, Any]],
    key: str,
    content_key: str = "content",
) -> None:
    if not evidences:
        return
    selected_index = st.selectbox(
        "Evidence",
        options=list(range(len(evidences))),
        format_func=lambda index: _evidence_label(evidences[index], index),
        key=key,
    )
    evidence = evidences[selected_index]
    st.caption(f"{evidence.get('source', 'unknown')} · {evidence.get('chunk_id', '-')}")
    st.write(str(evidence.get(content_key) or ""))


def _citation_label(citation: dict[str, Any]) -> str:
    return f"{citation.get('evidence_id', '-')} · {Path(str(citation.get('source', 'unknown'))).name}"


def _evidence_label(evidence: dict[str, Any], index: int) -> str:
    evidence_id = evidence.get("evidence_id") or evidence.get("chunk_id") or f"result_{index + 1}"
    return f"{evidence_id} · {Path(str(evidence.get('source', 'unknown'))).name}"


def _hybrid_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "Rank": item.get("rank"),
        "Evidence ID": item.get("evidence_id"),
        "Type": item.get("evidence_type"),
        "Fusion": item.get("fusion_score"),
        "Source": Path(str(item.get("source") or "unknown")).name,
        "Chunk": item.get("chunk_id"),
        "Content": _truncate(item.get("content")),
    }


def _vector_row(index: int, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "Rank": index,
        "Score": item.get("score"),
        "Source": Path(str(item.get("source") or "unknown")).name,
        "Chunk": item.get("chunk_id"),
        "Content": _truncate(item.get("content")),
    }


def _graph_row(index: int, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "Rank": index,
        "Source entity": item.get("source_name"),
        "Relation": item.get("relation_type"),
        "Target entity": item.get("target_name"),
        "Confidence": item.get("confidence"),
        "Document": Path(str(item.get("source") or "unknown")).name,
        "Evidence": _truncate(item.get("evidence")),
    }


def _truncate(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          html, body, [class*="css"] { letter-spacing: 0; }
          .block-container { max-width: 1440px; padding-top: 1.25rem; padding-bottom: 2rem; }
          h1 { font-size: 2rem !important; line-height: 1.15 !important; margin-bottom: 0.2rem !important; }
          h2, h3 { letter-spacing: 0 !important; }
          [data-testid="stSidebar"] { border-right: 1px solid #CBD5E1; }
          [data-testid="stMetric"] {
            background: #FFFFFF;
            border-left: 3px solid #0F766E;
            padding: 0.55rem 0.7rem;
          }
          .service-line {
            display: grid;
            grid-template-columns: 10px auto auto minmax(0, 1fr);
            align-items: center;
            gap: 0.45rem;
            min-height: 42px;
            padding: 0.45rem 0.65rem;
            background: #FFFFFF;
            border: 1px solid #D7DEE3;
            border-radius: 4px;
            margin-bottom: 0.45rem;
            font-size: 0.78rem;
          }
          .service-name { color: #334155; font-weight: 600; }
          .service-line strong { color: #172128; font-size: 0.72rem; }
          .service-detail {
            color: #64748B;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }
          .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #94A3B8; }
          .status-ready { background: #0F766E; }
          .status-unavailable { background: #B42318; }
          .status-not_configured { background: #B7791F; }
          div[data-testid="stDataFrame"] { border: 1px solid #D7DEE3; }
          @media (max-width: 900px) {
            .service-line { grid-template-columns: 10px auto auto; }
            .service-detail { grid-column: 2 / 4; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_styles()
presets, dataset_stats, demo_snapshot = load_demo_data()
if "active_analysis" not in st.session_state and demo_snapshot:
    st.session_state["active_analysis"] = demo_snapshot

with st.sidebar:
    st.header("查询配置")
    api_base_url = st.text_input("API 地址", value=DEFAULT_API_URL)
    if st.button("刷新状态", width="stretch"):
        fetch_readiness.clear()

    preset_options = [preset.question_id for preset in presets]
    selected_preset_id = st.selectbox(
        "预设问题",
        options=preset_options,
        format_func=lambda question_id: next(
            f"{preset.topic} · {preset.question}" for preset in presets if preset.question_id == question_id
        ),
        disabled=not presets,
    ) if presets else None

    selected_preset = next(
        (preset for preset in presets if preset.question_id == selected_preset_id),
        None,
    )
    if selected_preset and st.session_state.get("applied_preset_id") != selected_preset.question_id:
        st.session_state["question_input"] = selected_preset.question
        st.session_state["applied_preset_id"] = selected_preset.question_id

    question = st.text_area(
        "问题",
        key="question_input",
        height=110,
        placeholder="输入科研论文相关问题",
    )
    st.divider()
    vector_top_k = st.number_input("Vector TopK", min_value=1, max_value=20, value=3, step=1)
    graph_top_k = st.number_input("Graph TopK", min_value=1, max_value=30, value=5, step=1)
    graph_max_depth = st.number_input("Graph Depth", min_value=1, max_value=4, value=2, step=1)
    run_requested = st.button(
        "运行分析",
        type="primary",
        width="stretch",
        disabled=not question.strip(),
    )

readiness = None
readiness_error = None
try:
    readiness = fetch_readiness(api_base_url)
except (DemoApiError, ValueError) as exc:
    readiness_error = str(exc)

st.title("GraphRAG Research Workbench")
st.caption("科研论文混合检索与可解释问答")

render_service_status(readiness, readiness_error)

dataset_columns = st.columns(5)
dataset_columns[0].metric("数据集", dataset_stats.dataset_name)
dataset_columns[1].metric("文档", dataset_stats.document_count)
dataset_columns[2].metric("Chunks", dataset_stats.chunk_count)
dataset_columns[3].metric("实体", dataset_stats.entity_count)
dataset_columns[4].metric("关系", dataset_stats.relation_count)

if run_requested:
    with st.spinner("正在执行混合检索与问答..."):
        analysis_result = run_analysis(
            base_url=api_base_url,
            question=question.strip(),
            vector_top_k=int(vector_top_k),
            graph_top_k=int(graph_top_k),
            graph_max_depth=int(graph_max_depth),
        )
    if analysis_result.get("retrieval_debug") and analysis_result.get("qa"):
        st.session_state["last_successful_analysis"] = analysis_result
    if analysis_result.get("retrieval_debug") or analysis_result.get("qa"):
        st.session_state["active_analysis"] = analysis_result
    elif st.session_state.get("last_successful_analysis"):
        fallback_result = dict(st.session_state["last_successful_analysis"])
        fallback_result["errors"] = analysis_result.get("errors") or []
        fallback_result["stale"] = True
        st.session_state["active_analysis"] = fallback_result
    else:
        for error in analysis_result.get("errors") or []:
            st.error(error)

active_analysis = st.session_state.get("active_analysis")
if active_analysis:
    st.divider()
    st.caption(str(active_analysis.get("question") or ""))
    render_answer(active_analysis)
    render_comparison(active_analysis)
    render_evidence_tabs(active_analysis)
else:
    st.info("尚无分析结果。")
