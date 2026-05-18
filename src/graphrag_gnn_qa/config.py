from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "graphrag-gnn-qa"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    llm_provider: str = "deepseek"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"

    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "graphrag_neo4j_password"
    neo4j_database: str = "neo4j"

    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_db_name: str = "default"
    milvus_chunk_collection: str = "rag_chunks"
    milvus_node_collection: str = "graph_nodes"

    vector_top_k: int = 8
    graph_top_k: int = 8
    rerank_top_k: int = 5
    graph_max_depth: int = 2

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
