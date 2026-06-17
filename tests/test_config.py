import os

import pytest

from onaai.config import Config, load_config


def test_defaults():
    cfg = Config()
    assert cfg.model_path == "WeiboAI/VibeThinker-3B"
    assert cfg.backend == "transformers"
    assert cfg.top_k == -1


def test_validation_rejects_bad_backend():
    with pytest.raises(ValueError):
        Config(backend="not-a-backend")


def test_validation_rejects_bad_temperature():
    with pytest.raises(ValueError):
        Config(temperature=5.0)


def test_env_override(monkeypatch):
    monkeypatch.setenv("ONAAI_MODEL_PATH", "/local/models/vibethinker")
    monkeypatch.setenv("ONAAI_MAX_NEW_TOKENS", "1024")
    cfg = load_config()
    assert cfg.model_path == "/local/models/vibethinker"
    assert cfg.max_new_tokens == 1024  # coerced to int


def test_missing_config_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/to/config.yaml")
