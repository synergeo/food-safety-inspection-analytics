"""
Data loading utilities for the BVL Food Safety Inspection Analytics project.

The raw BVL 2023 monitoring dataset is a large semicolon-separated CSV file
encoded using Latin-1. These functions provide reusable loading utilities
for both initial inspection and full-data processing.
"""

from pathlib import Path

import pandas as pd


DEFAULT_SEPARATOR = ";"
DEFAULT_ENCODING = "latin1"


def load_bvl_sample(file_path, nrows=10):
    """
    Load a small sample of the raw BVL monitoring dataset.

    This is useful for inspecting the structure, column names, data types,
    and encoding before loading the complete dataset.

    Parameters
    ----------
    file_path : str or Path
        Path to the raw BVL CSV file.
    nrows : int, default=10
        Number of rows to load.

    Returns
    -------
    pandas.DataFrame
        Sample of the raw BVL dataset.
    """
    return pd.read_csv(
        Path(file_path),
        sep=DEFAULT_SEPARATOR,
        encoding=DEFAULT_ENCODING,
        nrows=nrows,
    )


def load_bvl_data(file_path, **kwargs):
    """
    Load the full raw BVL monitoring dataset.

    Additional pandas read_csv arguments can be supplied through kwargs,
    for example usecols or dtype when required.

    Parameters
    ----------
    file_path : str or Path
        Path to the raw BVL CSV file.
    **kwargs
        Additional arguments passed to pandas.read_csv.

    Returns
    -------
    pandas.DataFrame
        Loaded BVL monitoring data.
    """
    return pd.read_csv(
        Path(file_path),
        sep=DEFAULT_SEPARATOR,
        encoding=DEFAULT_ENCODING,
        **kwargs,
    )


def load_processed_data(file_path):
    """
    Load a processed project CSV file.

    Parameters
    ----------
    file_path : str or Path
        Path to the processed CSV file.

    Returns
    -------
    pandas.DataFrame
        Processed dataset.
    """
    return pd.read_csv(Path(file_path))
