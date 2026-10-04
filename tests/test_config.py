"""Unit tests for configuration management."""

import os
from pathlib import Path
import pytest
from business_entity_resolution.utils.config import (
    DEFAULT_S3_BUCKET,
    DEFAULT_S3_REGION,
    S3DataConfig,
)


def test_default_config():
    config = S3DataConfig()
    assert config.bucket == DEFAULT_S3_BUCKET
    assert config.region == DEFAULT_S3_REGION
    assert config.cache_dir.name == "raw"
    assert config.reports_dir.name == "reports"


def test_get_s3_key():
    config = S3DataConfig()
    key_s1 = config.get_s3_key("train", "source1")
    assert "train_source1.tsv" in key_s1

    key_gt = config.get_s3_key("train", "ground_truth")
    assert "train_ground_truth.tsv" in key_gt

    with pytest.raises(KeyError):
        config.get_s3_key("invalid_split", "source1")

    with pytest.raises(KeyError):
        config.get_s3_key("train", "invalid_dataset")


def test_get_local_path():
    config = S3DataConfig()
    path = config.get_local_path("train", "source1")
    assert path.name == "train_source1.tsv"
    assert path.parent == config.cache_dir


def test_env_override(monkeypatch):
    monkeypatch.setenv("ENTITY_RESOLUTION_S3_BUCKET", "custom-test-bucket")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("DATA_CACHE_DIR", "/tmp/custom_cache")

    config = S3DataConfig()
    assert config.bucket == "custom-test-bucket"
    assert config.region == "us-east-1"
    assert str(config.cache_dir).replace("\\", "/") == "/tmp/custom_cache"
