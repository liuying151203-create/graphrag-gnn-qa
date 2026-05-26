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
