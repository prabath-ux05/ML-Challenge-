"""I/O helpers for TSV reading and writing with OS-independent paths."""

import csv
from pathlib import Path
from typing import Any, Iterator, Optional, Union
import pandas as pd


def read_tsv(
    filepath_or_buffer: Union[str, Path, Any],
    columns: Optional[list] = None,
    dtype: Optional[Union[dict, type]] = str,
    nrows: Optional[int] = None,
    keep_default_na: bool = False,
    na_values: Optional[list] = None,
) -> pd.DataFrame:
    """Read a tab-separated value (TSV) file using explicit tab delimiters.

    Args:
        filepath_or_buffer: Path to file or file-like buffer.
        columns: Optional list of column names to load.
        dtype: Data type specification (defaults to str for all columns to avoid loss).
        nrows: Maximum number of rows to read.
        keep_default_na: Whether to convert default NA values like 'NA' or empty string.
                         Defaults to False to preserve raw entity strings.
        na_values: Additional strings to recognize as NA.

    Returns:
        pd.DataFrame containing the parsed TSV data.
    """
    return pd.read_csv(
        filepath_or_buffer,
        sep="\t",
        usecols=columns,
        dtype=dtype,
        nrows=nrows,
        keep_default_na=keep_default_na,
        na_values=na_values,
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
        low_memory=False,
    )


def read_tsv_chunks(
    filepath_or_buffer: Union[str, Path, Any],
    chunksize: int = 100_000,
    columns: Optional[list] = None,
    dtype: Optional[Union[dict, type]] = str,
    keep_default_na: bool = False,
) -> Iterator[pd.DataFrame]:
    """Read a TSV file in chunks using explicit tab delimiters.

    Args:
        filepath_or_buffer: Path to file or file-like buffer.
        chunksize: Number of lines per chunk.
        columns: Optional list of column names to load.
        dtype: Data type specification (defaults to str).
        keep_default_na: Whether to convert default NA values.

    Returns:
        Iterator of pd.DataFrame chunks.
    """
    yield from pd.read_csv(
        filepath_or_buffer,
        sep="\t",
        chunksize=chunksize,
        usecols=columns,
        dtype=dtype,
        keep_default_na=keep_default_na,
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
        low_memory=False,
    )


def write_tsv(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    columns: Optional[list] = None,
    index: bool = False,
) -> Path:
    """Write a DataFrame to a TSV file using explicit tab delimiters.

    Args:
        df: DataFrame to write.
        filepath: Target destination file path.
        columns: Optional subset of columns to write.
        index: Whether to write row names (index). Defaults to False.

    Returns:
        Path to the written file.
    """
    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        target,
        sep="\t",
        columns=columns,
        index=index,
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )
    return target
