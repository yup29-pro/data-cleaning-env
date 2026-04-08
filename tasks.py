# tasks.py
# ─────────────────────────────────────────────
# Task definitions for all 3 difficulty levels
# Each task has a description, dataset, grader
# ─────────────────────────────────────────────

import pandas as pd
from typing import Dict, Any, Tuple, Optional
from config import (
    TASK_EASY, TASK_MEDIUM, TASK_HARD, TASK_IDS, MAX_STEPS
)
from datasets import (
    generate_easy_dataset,
    generate_medium_dataset,
    generate_hard_dataset,
    detect_issues,
)
from graders import grade_task1, grade_task2, grade_task3
from models import TaskInfo


# ── Task Registry ─────────────────────────────

TASK_REGISTRY = {
    TASK_EASY:   {
        "task_id":     TASK_IDS[TASK_EASY],
        "difficulty":  TASK_EASY,
        "description": (
            "Fix a 10-row employee dataset: "
            "fill missing values in 'age' and 'salary', "
            "convert 'age' from string to integer."
        ),
        "max_steps":   MAX_STEPS,
        "issues": [
            "missing_values in age and salary columns",
            "wrong dtype: age should be int not string",
        ],
        "hints": [
            "Use fill_missing to handle null values",
            "Use fix_dtype to convert age to int",
            "Call submit when the dataset looks clean",
        ],
    },
    TASK_MEDIUM: {
        "task_id":     TASK_IDS[TASK_MEDIUM],
        "difficulty":  TASK_MEDIUM,
        "description": (
            "Fix a 20-row dataset: "
            "remove duplicate rows, "
            "handle salary outliers, "
            "standardize country values to 'United States' or 'UK'."
        ),
        "max_steps":   MAX_STEPS,
        "issues": [
            "duplicate rows present",
            "salary column has extreme outliers",
            "country has inconsistent values (USA, US, America, etc.)",
        ],
        "hints": [
            "Use drop_duplicates first",
            "Use remove_outliers on salary column",
            "Use standardize_values to unify country names",
        ],
    },
    TASK_HARD:   {
        "task_id":     TASK_IDS[TASK_HARD],
        "difficulty":  TASK_HARD,
        "description": (
            "Fix a 30-row dataset with all issues: "
            "nulls, duplicates, outliers, "
            "bad column names (trailing spaces, uppercase, special chars), "
            "inconsistent country values, "
            "and invalid dept_id values (referential integrity)."
        ),
        "max_steps":   MAX_STEPS,
        "issues": [
            "missing values in multiple columns",
            "bad column names: 'Full Name ', 'AGE', 'salary$', 'COUNTRY'",
            "salary outliers and negative values",
            "inconsistent country values",
            "invalid dept_id: 99 does not exist in lookup table",
        ],
        "hints": [
            "Start by renaming bad column names",
            "Then fix nulls, duplicates, and outliers",
            "Standardize country values",
            "Fix invalid dept_id values last",
        ],
    },
}


# ── Task Class ────────────────────────────────

class Task:
    """Represents a single task with its dataset and grader"""

    def __init__(self, difficulty: str):
        if difficulty not in TASK_REGISTRY:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        self.difficulty  = difficulty
        self.meta        = TASK_REGISTRY[difficulty]
        self.task_id     = self.meta["task_id"]
        self.description = self.meta["description"]
        self.max_steps   = self.meta["max_steps"]
        self.hints       = self.meta["hints"]

        # Dataset state
        self.original_df: Optional[pd.DataFrame] = None
        self.current_df:  Optional[pd.DataFrame] = None
        self.lookup_df:   Optional[pd.DataFrame] = None
        self.issues:      list = []

    def reset(self) -> pd.DataFrame:
        """Generate a fresh dataset and return it"""
        if self.difficulty == TASK_EASY:
            self.original_df = generate_easy_dataset()

        elif self.difficulty == TASK_MEDIUM:
            self.original_df = generate_medium_dataset()

        elif self.difficulty == TASK_HARD:
            self.original_df, self.lookup_df = generate_hard_dataset()

        self.current_df = self.original_df.copy()
        self.issues     = detect_issues(self.current_df, self.task_id)
        return self.current_df

    def grade(self) -> Tuple[float, Dict[str, Any]]:
        """Grade the current state of the dataset"""
        if self.current_df is None or self.original_df is None:
            return 0.0, {"error": "Task not initialized, call reset() first"}

        if self.difficulty == TASK_EASY:
            return grade_task1(self.original_df, self.current_df)

        elif self.difficulty == TASK_MEDIUM:
            return grade_task2(self.original_df, self.current_df)

        elif self.difficulty == TASK_HARD:
            return grade_task3(
                self.original_df, self.current_df, self.lookup_df
            )

        return 0.0, {}

    def update_issues(self):
        """Refresh issue list based on current state"""
        if self.current_df is not None:
            self.issues = detect_issues(self.current_df, self.task_id)

    def get_info(self) -> TaskInfo:
        """Return TaskInfo model"""
        score, _ = self.grade() if self.current_df is not None else (None, {})
        return TaskInfo(
            task_id     = self.task_id,
            difficulty  = self.difficulty,
            description = self.description,
            max_steps   = self.max_steps,
            score       = score,
        )


# ── Task Factory ──────────────────────────────

def get_task(difficulty: str) -> Task:
    """Create and return a Task instance"""
    return Task(difficulty)


def get_all_tasks() -> Dict[str, Task]:
    """Return all tasks as a dict"""
    return {d: Task(d) for d in [TASK_EASY, TASK_MEDIUM, TASK_HARD]}