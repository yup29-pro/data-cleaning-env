# graders.py
# ─────────────────────────────────────────────
# Grading logic for all 3 tasks
# Returns scores between 0.0 and 1.0
# ─────────────────────────────────────────────

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from utils import (
    compute_null_score,
    compute_duplicate_score,
    compute_dtype_score,
    detect_outliers_iqr,
    has_bad_column_names,
    clamp,
)


# ── Task 1 Grader — Easy ──────────────────────

def grade_task1(original_df: pd.DataFrame,
                current_df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    Scores:
    - 40% null values fixed
    - 40% correct dtypes (age=int, salary=float)
    - 20% no data loss (row count preserved)
    """
    details = {}

    # Null score (40%)
    null_score = compute_null_score(original_df, current_df)
    details["null_score"] = null_score

    # Dtype score (40%)
    expected_dtypes = {"age": "int64", "salary": "float64"}
    dtype_score = compute_dtype_score(current_df, expected_dtypes)
    details["dtype_score"] = dtype_score

    # Row preservation score (20%)
    expected_rows = len(original_df)
    actual_rows   = len(current_df)
    row_score = 1.0 if actual_rows == expected_rows else \
                clamp(actual_rows / expected_rows)
    details["row_score"] = row_score

    # Final weighted score
    final = (
        0.40 * null_score  +
        0.40 * dtype_score +
        0.20 * row_score
    )
    details["final"] = round(clamp(final), 4)
    return details["final"], details


# ── Task 2 Grader — Medium ────────────────────

def grade_task2(original_df: pd.DataFrame,
                current_df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    Scores:
    - 35% duplicates removed
    - 35% outliers handled in salary
    - 30% country values standardized
    """
    details = {}

    # Duplicate score (35%)
    dup_score = compute_duplicate_score(original_df, current_df)
    details["duplicate_score"] = dup_score

    # Outlier score (35%)
    if "salary" in current_df.columns:
        try:
            salary_col = pd.to_numeric(current_df["salary"], errors="coerce")
            outlier_mask    = detect_outliers_iqr(
                current_df.assign(salary=salary_col), "salary"
            )
            remaining_outliers = outlier_mask.sum()
            original_salary = pd.to_numeric(
                original_df["salary"], errors="coerce"
            )
            original_outliers = detect_outliers_iqr(
                original_df.assign(salary=original_salary), "salary"
            ).sum()
            if original_outliers == 0:
                outlier_score = 1.0
            else:
                fixed = original_outliers - remaining_outliers
                outlier_score = clamp(fixed / original_outliers)
        except Exception:
            outlier_score = 0.0
    else:
        outlier_score = 0.0
    details["outlier_score"] = outlier_score

    # Country standardization score (30%)
    if "country" in current_df.columns:
        standard_values = {"United States", "UK"}
        unique_vals     = set(current_df["country"].dropna().unique())
        non_standard    = unique_vals - standard_values
        country_score   = 1.0 if not non_standard else \
                          clamp(1 - len(non_standard) / len(unique_vals))
    else:
        country_score = 0.0
    details["country_score"] = country_score

    # Final weighted score
    final = (
        0.35 * dup_score     +
        0.35 * outlier_score +
        0.30 * country_score
    )
    details["final"] = round(clamp(final), 4)
    return details["final"], details


# ── Task 3 Grader — Hard ──────────────────────

def grade_task3(original_df: pd.DataFrame,
                current_df: pd.DataFrame,
                lookup_df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    Scores:
    - 25% null values fixed
    - 20% column names cleaned
    - 20% outliers handled
    - 20% country standardized
    - 15% referential integrity (dept_id valid)
    """
    details = {}

    # Null score (25%)
    null_score = compute_null_score(original_df, current_df)
    details["null_score"] = null_score

    # Column name score (20%)
    bad_cols = has_bad_column_names(current_df)
    col_score = 1.0 if not bad_cols else \
                clamp(1 - len(bad_cols) / len(current_df.columns))
    details["column_name_score"] = col_score

    # Outlier score (20%)
    numeric_cols = current_df.select_dtypes(include=[np.number]).columns
    salary_like  = [c for c in numeric_cols if "salary" in c.lower()]
    if salary_like:
        col = salary_like[0]
        try:
            outlier_mask = detect_outliers_iqr(current_df, col)
            orig_col     = [c for c in original_df.columns
                            if "salary" in c.lower()]
            if orig_col:
                orig_mask = detect_outliers_iqr(original_df, orig_col[0])
                orig_count = orig_mask.sum()
                if orig_count == 0:
                    outlier_score = 1.0
                else:
                    fixed = orig_count - outlier_mask.sum()
                    outlier_score = clamp(fixed / orig_count)
            else:
                outlier_score = 1.0
        except Exception:
            outlier_score = 0.0
    else:
        outlier_score = 0.0
    details["outlier_score"] = outlier_score

    # Country standardization (20%)
    country_cols = [c for c in current_df.columns if "country" in c.lower()]
    if country_cols:
        standard     = {"United States", "UK"}
        unique_vals  = set(current_df[country_cols[0]].dropna().unique())
        non_standard = unique_vals - standard
        country_score = 1.0 if not non_standard else \
                        clamp(1 - len(non_standard) / len(unique_vals))
    else:
        country_score = 0.0
    details["country_score"] = country_score

    # Referential integrity (15%)
    dept_cols = [c for c in current_df.columns if "dept_id" in c.lower()]
    if dept_cols and lookup_df is not None:
        valid_ids    = set(lookup_df["dept_id"].unique())
        actual_ids   = set(current_df[dept_cols[0]].dropna().unique())
        invalid      = actual_ids - valid_ids
        ref_score    = 1.0 if not invalid else \
                       clamp(1 - len(invalid) / len(actual_ids))
    else:
        ref_score = 0.0
    details["referential_integrity_score"] = ref_score

    # Final weighted score
    final = (
        0.25 * null_score    +
        0.20 * col_score     +
        0.20 * outlier_score +
        0.20 * country_score +
        0.15 * ref_score
    )
    details["final"] = round(clamp(final), 4)
    return details["final"], details