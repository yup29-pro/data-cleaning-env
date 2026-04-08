# datasets.py
# ─────────────────────────────────────────────
# CSV dataset generators for each difficulty
# Easy / Medium / Hard messy datasets
# ─────────────────────────────────────────────

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ── Task 1 — Easy ─────────────────────────────
def generate_easy_dataset() -> pd.DataFrame:
    """
    10 rows, issues:
    - Missing values in age & salary
    - Wrong dtype (age as string)
    """
    data = {
        "name": ["Aarav", "Priya", "Rohan", "Sneha", "Vikram",
                 "Ananya", "Kiran", "Arjun", "Meera", "Rahul"],
        "age":    ["25", "30", None, "22", "28",
                   "35", None, "40", "27", "33"],
        "salary": [50000, 60000, None, 45000, 55000,
                   None, 70000, 80000, None, 65000],
        "email":  [
            "aarav@mail.com", "priya@mail.com", "rohan@mail.com",
            "sneha@mail.com", "vikram@mail.com", "ananya@mail.com",
            "kiran@mail.com", "arjun@mail.com", "meera@mail.com",
            "rahul@mail.com"
        ]
    }
    return pd.DataFrame(data)


# ── Task 2 — Medium ───────────────────────────
def generate_medium_dataset() -> pd.DataFrame:
    """
    20 rows, issues:
    - Duplicate rows
    - Outliers in salary
    - Inconsistent country values
    """
    data = {
        "name": [
            "Aarav", "Priya", "Rohan", "Sneha", "Vikram",
            "Ananya", "Kiran", "Arjun", "Meera", "Rahul",
            "Aarav", "Priya", "Pooja", "Aditya", "Nisha",
            "Divya", "Suresh", "Amit", "Kavya", "Riya"
        ],
        "age": [25, 30, 35, 22, 28, 40, 27, 33, 29, 31,
                25, 30, 26, 38, 24, 32, 36, 28, 30, 27],
        "salary": [
            50000, 60000, 55000, 45000, 58000,
            62000, 48000, 70000, 999999, 65000,
            50000, 60000, 47000, 72000, 43000,
            -5000, 68000, 54000, 61000, 49000
        ],
        "country": [
            "USA", "United States", "US", "USA", "America",
            "UK", "United Kingdom", "UK", "USA", "US",
            "USA", "United States", "UK", "USA", "US",
            "USA", "UK", "United Kingdom", "USA", "US"
        ]
    }
    return pd.DataFrame(data)


# ── Task 3 — Hard ─────────────────────────────
def generate_hard_dataset() -> tuple:
    """
    30 rows, issues:
    - All of easy + medium issues
    - Bad column names
    - Mixed dtype columns
    Returns (main_df, lookup_df) for referential integrity check
    """
    main_data = {
        "Full Name ": [
            "Aarav", "Priya", "Rohan", "Sneha", "Vikram",
            "Ananya", "Kiran", "Arjun", "Meera", "Rahul",
            "Pooja", "Aditya", "Nisha", "Divya", "Suresh",
            "Amit", "Kavya", "Riya", "Sanjay", "Deepa",
            "Rajesh", "Sunita", "Manoj", "Lakshmi", "Ganesh",
            "Bharti", "Aarav", "Priya", "Rohan", "Sneha"
        ],
        "AGE": [
            "25", "30", None, "22", "28",
            "35", "27", "40", "abc", "33",
            "26", "38", "24", "32", "36",
            "28", "30", "27", "29", "31",
            "25", "30", None, "22", "28",
            "35", "25", "30", "35", "22"
        ],
        "salary$": [
            50000, 60000, None, 45000, 55000,
            None, 70000, 80000, None, 65000,
            47000, 72000, 43000, -5000, 68000,
            54000, 61000, 49000, 999999, 58000,
            50000, 60000, 55000, 45000, 58000,
            62000, 50000, 60000, 55000, 45000
        ],
        "COUNTRY": [
            "USA", "United States", "US", "USA", "America",
            "UK", "United Kingdom", "UK", "USA", "US",
            "USA", "United States", "UK", "USA", "US",
            "USA", "UK", "United Kingdom", "USA", "US",
            "USA", "United States", "US", "USA", "America",
            "UK", "USA", "US", "UK", "USA"
        ],
        "dept_id": [
            1, 2, 3, 1, 2, 3, 1, 2, 3, 1,
            2, 3, 1, 2, 99, 1, 2, 3, 1, 2,
            3, 1, 2, 3, 1, 2, 3, 1, 2, 3
        ]
    }

    lookup_data = {
        "dept_id":   [1, 2, 3],
        "dept_name": ["Engineering", "Marketing", "Sales"]
    }

    return pd.DataFrame(main_data), pd.DataFrame(lookup_data)


# ── Issue Detection ───────────────────────────
def detect_issues(df: pd.DataFrame, task_id: str) -> list:
    """Detect all issues in a dataframe"""
    issues = []

    # Null values
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            issues.append(f"missing_values::{col}::{count}")

    # Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append(f"duplicate_rows::{dup_count}")

    # Outliers (numeric columns)
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5 * IQR) |
                    (df[col] > Q3 + 1.5 * IQR)).sum()
        if outliers > 0:
            issues.append(f"outliers::{col}::{outliers}")

    # Bad column names
    for col in df.columns:
        if col != col.strip() or col != col.lower() or \
           any(c in col for c in ["$", "@", "#", "!"]):
            issues.append(f"bad_column_name::{col}")

    return issues