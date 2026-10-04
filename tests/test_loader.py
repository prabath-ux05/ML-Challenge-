"""Unit tests for S3DataLoader with mocked S3 client."""

from unittest.mock import MagicMock
import pandas as pd
import pytest
from business_entity_resolution.data.loader import S3DataLoader
from business_entity_resolution.utils.config import S3DataConfig


def test_s3_data_loader_initialization():
    mock_s3 = MagicMock()
    config = S3DataConfig()
    loader = S3DataLoader(config=config, s3_client=mock_s3)
    assert loader.s3_client == mock_s3
    assert loader.config == config


def test_s3_data_loader_download_dataset(tmp_path):
    mock_s3 = MagicMock()
    config = S3DataConfig(cache_dir=tmp_path)
    loader = S3DataLoader(config=config, s3_client=mock_s3)

    target_file = tmp_path / "train_source1.tsv"

    # Simulate download_file creating file
    def mock_download(Bucket, Key, Filename, Config):
        with open(Filename, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")

    mock_s3.download_file.side_effect = mock_download

    path = loader.download_dataset("train", "source1")
    assert path == target_file
    assert path.exists()
    assert mock_s3.download_file.called


def test_load_ground_truth_invalid_split():
    mock_s3 = MagicMock()
    loader = S3DataLoader(s3_client=mock_s3)
    with pytest.raises(ValueError):
        loader.load_ground_truth(split="test")
