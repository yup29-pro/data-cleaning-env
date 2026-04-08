# utils.py
# ─────────────────────────────────────────────
# Helper functions used across the project
# ─────────────────────────────────────────────

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple


# ── DataFrame Helpers ─────────────────────────

def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame to list of dicts (for JSON serialization)"""
    return df.where(pd.notnull(df), None).to_dict(orient="records")


def df_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a quick summary of a DataFrame"""
    return {
        "shape":          list(df.shape),
        "columns":        list(df.columns),
        "dtypes":         {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts":    df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def get_null_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Return null count per column"""
    return {col: int(count) for col, count in df.isnull().sum().items()}


def get_duplicate_count(df: pd.DataFrame) -> int:
    """Return number of duplicate rows"""
    return int(df.duplicated().sum())


# ── Outlier Helpers ───────────────────────────

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """Return boolean mask of outliers using IQR method"""
    Q1  = df[column].quantile(0.25)
    Q3  = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (df[column] < lower) | (df[column] > upper)


def detect_outliers_zscore(df: pd.DataFrame, column: str,
                            threshold: float = 3.0) -> pd.Series:
    """Return boolean mask of outliers using Z-score method"""
    mean = df[column].mean()
    std  = df[column].std()
    if std == 0:
        return pd.Series([False] * len(df))
    return ((df[column] - mean) / std).abs() > threshold


# ── Column Name Helpers ───────────────────────

def clean_column_name(name: str) -> str:
    """Normalize a column name to snake_case"""
    import re
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def has_bad_column_names(df: pd.DataFrame) -> List[str]:
    """Return list of columns with bad names"""
    bad = []
    for col in df.columns:
        if col != clean_column_name(col):
            bad.append(col)
    return bad


# ── Value Standardization ─────────────────────

def standardize_column(df: pd.DataFrame, column: str,
                        mapping: Dict[str, str]) -> pd.DataFrame:
    """Replace values in a column using a mapping dict"""
    df = df.copy()
    df[column] = df[column].replace(mapping)
    return df


def get_value_counts(df: pd.DataFrame, column: str) -> Dict[str, int]:
    """Return value counts for a column"""
    return df[column].value_counts().to_dict()


# ── Scoring Helpers ───────────────────────────

def compute_null_score(original_df: pd.DataFrame,
                        current_df: pd.DataFrame) -> float:
    """Score based on how many nulls have been fixed (0.0 to 1.0)"""
    original_nulls = original_df.isnull().sum().sum()
    if original_nulls == 0:
        return 1.0
    current_nulls = current_df.isnull().sum().sum()
    fixed = original_nulls - current_nulls
    return round(max(0.0, fixed / original_nulls), 4)


def compute_duplicate_score(original_df: pd.DataFrame,
                              current_df: pd.DataFrame) -> float:
    """Score based on how many duplicates have been removed (0.0 to 1.0)"""
    original_dups = original_df.duplicated().sum()
    if original_dups == 0:
        return 1.0
    current_dups = current_df.duplicated().sum()
    fixed = original_dups - current_dups
    return round(max(0.0, fixed / original_dups), 4)


def compute_dtype_score(df: pd.DataFrame,
                         expected: Dict[str, str]) -> float:
    """Score based on how many columns have correct dtype"""
    if not expected:
        return 1.0
    correct = sum(
        1 for col, dtype in expected.items()
        if col in df.columns and str(df[col].dtype) == dtype
    )
    return round(correct / len(expected), 4)


# ── Misc ──────────────────────────────────────

def clamp(value: float, min_val: float = 0.0,
          max_val: float = 1.0) -> float:
    """Clamp a float between min and max"""
    return max(min_val, min(max_val, value))


def format_score(score: float) -> str:
    """Format score for display"""
    return f"{score:.3f}"