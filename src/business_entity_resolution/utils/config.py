"""Configuration management and file path definitions."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

# Project directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

# Default S3 Bucket and keys
DEFAULT_S3_BUCKET = "business-entity-resolution-data-2026"
DEFAULT_S3_REGION = "ap-southeast-1"

# S3 dataset keys
S3_KEYS: Dict[str, Dict[str, str]] = {
    "train": {
        "source1": "business-entity-resolution/train/train/train_source1.tsv",
        "source2": "business-entity-resolution/train/train/train_source2.tsv",
        "source3": "business-entity-resolution/train/train/train_source3.tsv",
        "ground_truth": "business-entity-resolution/train/train/train_ground_truth.tsv",
    },
    "test": {
        "source1": "business-entity-resolution/test/test/test_source1.tsv",
        "source2": "business-entity-resolution/test/test/test_source2.tsv",
        "source3": "business-entity-resolution/test/test/test_source3.tsv",
    },
}


@dataclass
class S3DataConfig:
    """Configuration for S3 data loading and local caching.

    AWS credentials are intentionally omitted: credentials must come directly from
    the AWS environment (environment variables, AWS config, IAM role, etc.),
    never hardcoded.
    """

    bucket: str = field(
        default_factory=lambda: os.getenv("ENTITY_RESOLUTION_S3_BUCKET", DEFAULT_S3_BUCKET)
    )
    region: Optional[str] = field(
        default_factory=lambda: os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", DEFAULT_S3_REGION))
    )
    cache_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_CACHE_DIR", str(DEFAULT_DATA_DIR)))
    )
    reports_dir: Path = field(
        default_factory=lambda: Path(os.getenv("REPORTS_DIR", str(DEFAULT_REPORTS_DIR)))
    )
    s3_keys: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: dict(S3_KEYS)
    )

    def get_s3_key(self, split: str, dataset_name: str) -> str:
        """Get the S3 object key for a specific split and dataset.

        Args:
            split: 'train' or 'test'.
            dataset_name: 'source1', 'source2', 'source3', or 'ground_truth'.

        Returns:
            S3 object key string.
        """
        if split not in self.s3_keys:
            raise KeyError(f"Unknown split: '{split}'. Expected one of {list(self.s3_keys.keys())}")
        if dataset_name not in self.s3_keys[split]:
            raise KeyError(
                f"Unknown dataset '{dataset_name}' for split '{split}'. "
                f"Expected one of {list(self.s3_keys[split].keys())}"
            )
        return self.s3_keys[split][dataset_name]

    def get_local_path(self, split: str, dataset_name: str) -> Path:
        """Get the local cache file path for a dataset.

        Args:
            split: 'train' or 'test'.
            dataset_name: 'source1', 'source2', 'source3', or 'ground_truth'.

        Returns:
            Path object pointing to local TSV file.
        """
        filename = f"{split}_{dataset_name}.tsv"
        return self.cache_dir / filename
