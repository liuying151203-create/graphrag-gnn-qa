from dataclasses import dataclass

from graphrag_gnn_qa.retrieval.graph_retriever import RetrievedGraphRelation
from graphrag_gnn_qa.retrieval.hybrid_result import HybridEvidence
from graphrag_gnn_qa.retrieval.vector_retriever import RetrievedChunk


VECTOR_EMPTY_CONTEXT = "No vector context retrieved."
GRAPH_EMPTY_CONTEXT = "No graph context retrieved."
HYBRID_EMPTY_CONTEXT = "No hybrid evidence retrieved."


@dataclass(frozen=True)
class GraphRAGContext:
    vector_context: str
    graph_context: str


@dataclass(frozen=True)
class HybridRAGContext:
    hybrid_context: str


def build_vector_context(chunks: list[RetrievedChunk]) -> str:
    vector_context = "\n\n".join(
        f"[Source {index}] chunk_id={chunk.chunk_id} score={chunk.score:.4f}\n{chunk.content}"
        for index, chunk in enumerate(chunks, start=1)
    )
    if not vector_context:
        return VECTOR_EMPTY_CONTEXT
    return vector_context


def build_graph_context(graph_relations: list[RetrievedGraphRelation] | None = None) -> str:
    graph_context = "\n\n".join(
        f"[Graph Source {index}] "
        f"center={relation.center_name} ({relation.center_type}) "
        f"triple={relation.source_name} ({relation.source_type}) "
        f"-[:{relation.relation_type}]-> "
        f"{relation.target_name} ({relation.target_type}) "
        f"chunk_id={relation.chunk_id} confidence={relation.confidence:.4f}\n"
        f"evidence={relation.evidence}"
        for index, relation in enumerate(graph_relations or [], start=1)
    )
    if not graph_context:
        return GRAPH_EMPTY_CONTEXT
    return graph_context


def build_hybrid_context(hybrid_evidences: list[HybridEvidence]) -> str:
    hybrid_context = "\n\n".join(
        f"[Hybrid Evidence {index}] "
        f"evidence_id={evidence.evidence_id} "
        f"type={evidence.evidence_type.value} "
        f"rank={evidence.rank} "
        f"score={evidence.score:.4f} "
        f"fusion_score={evidence.fusion_score:.4f} "
        f"document_id={evidence.document_id} "
        f"chunk_id={evidence.chunk_id} "
        f"source={evidence.source}\n"
        f"{evidence.content}"
        for index, evidence in enumerate(hybrid_evidences, start=1)
    )
    if not hybrid_context:
        return HYBRID_EMPTY_CONTEXT
    return hybrid_context


def build_graphrag_context(
    chunks: list[RetrievedChunk],
    graph_relations: list[RetrievedGraphRelation] | None = None,
) -> GraphRAGContext:
    return GraphRAGContext(
        vector_context=build_vector_context(chunks),
        graph_context=build_graph_context(graph_relations),
    )


def build_hybrid_graphrag_context(hybrid_evidences: list[HybridEvidence]) -> HybridRAGContext:
    return HybridRAGContext(hybrid_context=build_hybrid_context(hybrid_evidences))


def build_rag_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    graph_relations: list[RetrievedGraphRelation] | None = None,
) -> str:
    context = build_graphrag_context(chunks=chunks, graph_relations=graph_relations)
    return (
        "Use the following retrieved vector context and graph context to answer the question.\n"
        "If the context does not contain enough information, say that the current documents do not provide enough evidence.\n"
        "Cite relevant vector source numbers and graph source numbers in the answer when possible.\n\n"
        f"Vector Context:\n{context.vector_context}\n\n"
        f"Graph Context:\n{context.graph_context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


def build_hybrid_rag_prompt(
    question: str,
    hybrid_evidences: list[HybridEvidence],
) -> str:
    context = build_hybrid_graphrag_context(hybrid_evidences=hybrid_evidences)
    return (
        "Use the following deduplicated hybrid retrieval evidence to answer the question.\n"
        "If the evidence does not contain enough information, say that the current documents do not provide enough evidence.\n"
        "Cite relevant hybrid evidence IDs in the answer when possible.\n\n"
        f"Hybrid Evidence Context:\n{context.hybrid_context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
