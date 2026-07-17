from dataclasses import dataclass

from graphrag_gnn_qa.config import Settings
from graphrag_gnn_qa.graph.neo4j_store import Neo4jGraphStore
from graphrag_gnn_qa.llm.client import OpenAICompatibleLLMClient
from graphrag_gnn_qa.rag.qa_service import RAGQAService
from graphrag_gnn_qa.rerank import (
    BGEEvidenceReranker,
    FallbackEvidenceReranker,
    KeywordOverlapEvidenceReranker,
)
from graphrag_gnn_qa.rerank.evidence_reranker import EvidenceReranker
from graphrag_gnn_qa.retrieval.graph_retriever import GraphRetriever
from graphrag_gnn_qa.retrieval.vector_retriever import VectorRetriever
from graphrag_gnn_qa.vectorstore.embedding import SentenceTransformerEmbeddingModel
from graphrag_gnn_qa.vectorstore.milvus_client import MilvusVectorStore


@dataclass
class RuntimeResources:
    vector_store: MilvusVectorStore
    graph_store: Neo4jGraphStore
    vector_retriever: VectorRetriever
    graph_retriever: GraphRetriever
    reranker: EvidenceReranker
    qa_service: RAGQAService | None

    def close(self) -> None:
        try:
            self.graph_store.close()
        finally:
            self.vector_store.close()


def build_evidence_reranker(settings: Settings) -> EvidenceReranker:
    keyword_reranker = KeywordOverlapEvidenceReranker()
    if settings.reranker_type == "keyword":
        return keyword_reranker
    if settings.reranker_type == "bge":
        return FallbackEvidenceReranker(
            primary=BGEEvidenceReranker(model_name=settings.reranker_model),
            fallback=keyword_reranker,
        )
    raise ValueError(f"Unsupported reranker type: {settings.reranker_type}")


def build_runtime_resources(settings: Settings) -> RuntimeResources:
    embedding_model = SentenceTransformerEmbeddingModel(model_name=settings.embedding_model)
    vector_store = MilvusVectorStore(
        host=settings.milvus_host,
        port=settings.milvus_port,
        collection_name=settings.milvus_chunk_collection,
    )
    graph_store = None
    try:
        vector_store.connect()
        graph_store = Neo4jGraphStore(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        vector_retriever = VectorRetriever(embedding_model=embedding_model, vector_store=vector_store)
        graph_retriever = GraphRetriever(graph_store=graph_store)
        reranker = build_evidence_reranker(settings)
        qa_service = _build_qa_service(
            settings=settings,
            vector_retriever=vector_retriever,
            graph_retriever=graph_retriever,
            reranker=reranker,
        )
        return RuntimeResources(
            vector_store=vector_store,
            graph_store=graph_store,
            vector_retriever=vector_retriever,
            graph_retriever=graph_retriever,
            reranker=reranker,
            qa_service=qa_service,
        )
    except Exception:
        try:
            if graph_store is not None:
                graph_store.close()
        finally:
            vector_store.close()
        raise


def _build_qa_service(
    settings: Settings,
    vector_retriever: VectorRetriever,
    graph_retriever: GraphRetriever,
    reranker: EvidenceReranker,
) -> RAGQAService | None:
    if not settings.llm_api_key:
        return None

    llm_client = OpenAICompatibleLLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )
    return RAGQAService(
        retriever=vector_retriever,
        llm_client=llm_client,
        graph_retriever=graph_retriever,
        graph_top_k=settings.graph_top_k,
        graph_max_depth=settings.graph_max_depth,
        fusion_score_weight=settings.fusion_score_weight,
        fusion_rank_weight=settings.fusion_rank_weight,
        reranker=reranker,
        rerank_top_k=settings.rerank_top_k,
    )
