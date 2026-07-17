import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EmbeddingRecord:
    chunk_id: str
    document_id: str
    content: str
    source: str
    file_name: str
    file_type: str
    embedding: list[float]


def read_embedding_records(file_path: Path) -> list[EmbeddingRecord]:
    if not file_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {file_path}")

    records = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw_record = json.loads(line)
        records.append(
            EmbeddingRecord(
                chunk_id=raw_record["chunk_id"],
                document_id=raw_record["document_id"],
                content=raw_record["content"],
                source=raw_record["source"],
                file_name=raw_record["file_name"],
                file_type=raw_record["file_type"],
                embedding=[float(value) for value in raw_record["embedding"]],
            )
        )
    return records


def infer_embedding_dimension(records: list[EmbeddingRecord]) -> int:
    if not records:
        raise ValueError("Cannot infer embedding dimension from empty records")

    dimension = len(records[0].embedding)
    if dimension == 0:
        raise ValueError("Embedding dimension must be greater than 0")

    for record in records:
        if len(record.embedding) != dimension:
            raise ValueError("All embeddings must have the same dimension")

    return dimension


def prepare_insert_columns(records: list[EmbeddingRecord]) -> list[list[Any]]:
    return [
        [record.chunk_id for record in records],
        [record.document_id for record in records],
        [record.content for record in records],
        [record.source for record in records],
        [record.file_name for record in records],
        [record.file_type for record in records],
        [record.embedding for record in records],
    ]


class MilvusVectorStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "rag_chunks",
        alias: str = "default",
    ) -> None:
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.alias = alias

    def connect(self) -> None:
        from pymilvus import connections

        connections.connect(alias=self.alias, host=self.host, port=str(self.port))

    def close(self) -> None:
        from pymilvus import connections

        connections.disconnect(self.alias)

    def ping(self) -> None:
        from pymilvus import utility

        utility.list_collections(using=self.alias)

    def create_collection(self, dimension: int, drop_existing: bool = False) -> None:
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

        if drop_existing and utility.has_collection(self.collection_name, using=self.alias):
            utility.drop_collection(self.collection_name, using=self.alias)

        if utility.has_collection(self.collection_name, using=self.alias):
            return

        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="file_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        ]
        schema = CollectionSchema(fields=fields, description="RAG chunk embeddings")
        collection = Collection(name=self.collection_name, schema=schema, using=self.alias)
        collection.create_index(
            field_name="embedding",
            index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}},
        )

    def insert_records(self, records: list[EmbeddingRecord]) -> int:
        if not records:
            return 0

        from pymilvus import Collection

        collection = Collection(name=self.collection_name, using=self.alias)
        mutation_result = collection.insert(prepare_insert_columns(records))
        collection.flush()
        return int(mutation_result.insert_count)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        from pymilvus import Collection

        collection = Collection(name=self.collection_name, using=self.alias)
        collection.load()
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {}},
            limit=top_k,
            output_fields=["chunk_id", "document_id", "content", "source", "file_name", "file_type"],
        )

        return [
            {
                "score": float(hit.score),
                "chunk_id": hit.entity.get("chunk_id"),
                "document_id": hit.entity.get("document_id"),
                "content": hit.entity.get("content"),
                "source": hit.entity.get("source"),
                "file_name": hit.entity.get("file_name"),
                "file_type": hit.entity.get("file_type"),
            }
            for hit in results[0]
        ]
