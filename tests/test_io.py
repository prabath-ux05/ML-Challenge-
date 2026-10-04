"""Unit tests for I/O helpers."""

import io
import pandas as pd
from business_entity_resolution.utils.io import read_tsv, read_tsv_chunks, write_tsv


def test_write_and_read_tsv(tmp_path):
    filepath = tmp_path / "test.tsv"
    df = pd.DataFrame({
        "entity_id": ["E-1", "E-2"],
        "business_name": ["Foo Bar", "Baz Qux"],
        "business_address": ["100 St", "200 Ave"],
        "country": ["US", "India"],
    })

    write_tsv(df, filepath)
    assert filepath.exists()

    loaded = read_tsv(filepath)
    assert loaded.shape == (2, 4)
    assert loaded["business_name"].tolist() == ["Foo Bar", "Baz Qux"]


def test_read_tsv_chunks(tmp_path):
    filepath = tmp_path / "test_chunks.tsv"
    df = pd.DataFrame({
        "entity_id": [f"E-{i}" for i in range(10)],
        "val": [f"v{i}" for i in range(10)],
    })
    write_tsv(df, filepath)

    chunks = list(read_tsv_chunks(filepath, chunksize=3))
    assert len(chunks) == 4
    assert len(chunks[0]) == 3
    assert len(chunks[-1]) == 1
