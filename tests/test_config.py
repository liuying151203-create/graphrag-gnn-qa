import pytest
from pydantic import ValidationError

from graphrag_gnn_qa.config import Settings


def test_settings_default_fusion_weights(monkeypatch) -> None:
    monkeypatch.delenv("FUSION_SCORE_WEIGHT", raising=False)
    monkeypatch.delenv("FUSION_RANK_WEIGHT", raising=False)

    settings = Settings(_env_file=None)

    assert settings.fusion_score_weight == 0.7
    assert settings.fusion_rank_weight == 0.3


def test_settings_rejects_negative_fusion_weight() -> None:
    with pytest.raises(ValidationError):
        Settings(fusion_score_weight=-1, _env_file=None)


def test_settings_rejects_zero_total_fusion_weight() -> None:
    with pytest.raises(ValidationError):
        Settings(fusion_score_weight=0, fusion_rank_weight=0, _env_file=None)


def test_settings_rejects_invalid_rerank_top_k() -> None:
    with pytest.raises(ValidationError):
        Settings(rerank_top_k=0, _env_file=None)


def test_settings_default_reranker_type() -> None:
    settings = Settings(_env_file=None)

    assert settings.reranker_type == "keyword"


def test_settings_accepts_bge_reranker_type() -> None:
    settings = Settings(reranker_type="bge", _env_file=None)

    assert settings.reranker_type == "bge"


def test_settings_rejects_invalid_reranker_type() -> None:
    with pytest.raises(ValidationError):
        Settings(reranker_type="cross_encoder", _env_file=None)


def test_settings_default_document_ingestion_options() -> None:
    settings = Settings(_env_file=None)

    assert settings.document_upload_max_bytes == 20 * 1024 * 1024
    assert settings.ingestion_chunk_size == 800
    assert settings.ingestion_chunk_overlap == 120
    assert settings.ingestion_embedding_batch_size == 16


def test_settings_rejects_invalid_ingestion_chunk_overlap() -> None:
    with pytest.raises(ValidationError):
        Settings(
            ingestion_chunk_size=100,
            ingestion_chunk_overlap=100,
            _env_file=None,
        )


def test_settings_rejects_chunk_size_larger_than_milvus_field() -> None:
    with pytest.raises(ValidationError):
        Settings(ingestion_chunk_size=8193, _env_file=None)
