"""Unit tests for data schemas."""

import pandas as pd
import pytest
from business_entity_resolution.data.schema import (
    COL_BUSINESS_ADDRESS,
    COL_BUSINESS_NAME,
    COL_COUNTRY,
    COL_ENTITY_ID,
    COL_MATCHED_ENTITY_IDS,
    COL_SOURCE1_ENTITY_ID,
    validate_ground_truth_schema,
    validate_source_schema,
)


def test_validate_source_schema_valid():
    df = pd.DataFrame({
        COL_ENTITY_ID: ["S1-1"],
        COL_BUSINESS_NAME: ["Acme Corp"],
        COL_BUSINESS_ADDRESS: ["123 Main St"],
        COL_COUNTRY: ["US"],
    })
    assert validate_source_schema(df) is True


def test_validate_source_schema_invalid():
    df = pd.DataFrame({
        COL_ENTITY_ID: ["S1-1"],
        COL_BUSINESS_NAME: ["Acme Corp"],
    })
    with pytest.raises(ValueError):
        validate_source_schema(df)

    assert validate_source_schema(df, raise_error=False) is False


def test_validate_ground_truth_schema_valid():
    df = pd.DataFrame({
        COL_SOURCE1_ENTITY_ID: ["S1-1"],
        COL_MATCHED_ENTITY_IDS: ["S2-1,S3-1"],
    })
    assert validate_ground_truth_schema(df) is True


def test_validate_ground_truth_schema_invalid():
    df = pd.DataFrame({
        COL_SOURCE1_ENTITY_ID: ["S1-1"],
    })
    with pytest.raises(ValueError):
        validate_ground_truth_schema(df)

    assert validate_ground_truth_schema(df, raise_error=False) is False
