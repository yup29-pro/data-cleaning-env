# environment.py
# ─────────────────────────────────────────────
# Core OpenEnv class — step / reset / state
# Full OpenEnv spec compliant
# ─────────────────────────────────────────────

import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple

from config import (
    MAX_STEPS,
    REWARD_ISSUE_FIXED,
    REWARD_SUBMIT_BONUS,
    PENALTY_WRONG_ACTION,
    PENALTY_DESTRUCTIVE,
    TASK_EASY, TASK_MEDIUM, TASK_HARD,
)
from models import Observation, Action, Reward, StepResult
from tasks import Task, get_task
from utils import (
    df_to_records,
    get_null_counts,
    get_duplicate_count,
    detect_outliers_iqr,
    clean_column_name,
    standardize_column,
    clamp,
)
from datasets import detect_issues


class DataCleaningEnv:
    """
    OpenEnv-compliant Data Cleaning Environment.

    The agent receives a messy CSV dataset and must
    issue cleaning actions to fix it step by step.

    Actions:
        fill_missing        — fill null values
        drop_duplicates     — remove duplicate rows
        fix_dtype           — convert column dtype
        rename_column       — rename a column
        remove_outliers     — remove outlier rows
        standardize_values  — replace values via mapping
        submit              — finalize and trigger grader
    """

    def __init__(self, difficulty: str = TASK_EASY):
        self.difficulty    = difficulty
        self.task: Task    = get_task(difficulty)
        self.step_count    = 0
        self.total_reward  = 0.0
        self.done          = False
        self._prev_score   = 0.0
        self._initialized  = False

    # ── OpenEnv API ───────────────────────────

    def reset(self) -> Observation:
        """Start a fresh episode"""
        self.task.reset()
        self.step_count   = 0
        self.total_reward = 0.0
        self.done         = False
        self._prev_score  = 0.0
        self._initialized = True
        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        """Execute one action and return result"""
        if not self._initialized:
            self.reset()

        if self.done:
            obs = self._make_observation()
            reward = Reward(value=0.0, total=self.total_reward,
                           reason="Episode already done")
            return StepResult(observation=obs, reward=reward,
                            done=True, info={})

        self.step_count += 1
        reward_value, reason, info = self._execute_action(action)

        # Clamp and accumulate
        reward_value       = clamp(reward_value, -1.0, 1.0)
        self.total_reward  = round(self.total_reward + reward_value, 4)

        # Check done conditions
        if self.step_count >= MAX_STEPS:
            self.done = True
            reason   += " | Max steps reached"

        reward = Reward(
            value  = round(reward_value, 4),
            total  = self.total_reward,
            reason = reason,
        )
        obs = self._make_observation()
        obs.done = self.done

        return StepResult(
            observation = obs,
            reward      = reward,
            done        = self.done,
            info        = info,
        )

    def state(self) -> Dict[str, Any]:
        """Return current environment state as dict"""
        if not self._initialized:
            return {"status": "not initialized, call reset() first"}

        score, details = self.task.grade()
        return {
            "task_id":        self.task.task_id,
            "difficulty":     self.difficulty,
            "step":           self.step_count,
            "max_steps":      MAX_STEPS,
            "total_reward":   self.total_reward,
            "current_score":  score,
            "done":           self.done,
            "issues":         self.task.issues,
            "grade_details":  details,
            "dataframe":      df_to_records(self.task.current_df),
            "columns":        list(self.task.current_df.columns),
            "null_counts":    get_null_counts(self.task.current_df),
            "duplicate_rows": get_duplicate_count(self.task.current_df),
        }

    # ── Action Executor ───────────────────────

    def _execute_action(self, action: Action
                        ) -> Tuple[float, str, Dict]:
        """Route action to handler, return (reward, reason, info)"""
        df   = self.task.current_df
        p    = action.parameters
        info = {}

        try:
            if action.action_type == "fill_missing":
                df, reward, reason = self._fill_missing(df, p)

            elif action.action_type == "drop_duplicates":
                df, reward, reason = self._drop_duplicates(df)

            elif action.action_type == "fix_dtype":
                df, reward, reason = self._fix_dtype(df, p)

            elif action.action_type == "rename_column":
                df, reward, reason = self._rename_column(df, p)

            elif action.action_type == "remove_outliers":
                df, reward, reason = self._remove_outliers(df, p)

            elif action.action_type == "standardize_values":
                df, reward, reason = self._standardize_values(df, p)

            elif action.action_type == "submit":
                df, reward, reason, info = self._submit(df)

            else:
                reward = PENALTY_WRONG_ACTION
                reason = f"Unknown action: {action.action_type}"

        except Exception as e:
            reward = PENALTY_WRONG_ACTION
            reason = f"Action failed: {str(e)}"
            info   = {"error": str(e)}

        self.task.current_df = df
        self.task.update_issues()
        return reward, reason, info

    # ── Action Handlers ───────────────────────

    def _fill_missing(self, df: pd.DataFrame,
                      params: Dict) -> Tuple[pd.DataFrame, float, str]:
        column   = params.get("column")
        strategy = params.get("strategy", "mean")

        if column not in df.columns:
            return df, PENALTY_WRONG_ACTION, f"Column '{column}' not found"

        null_before = df[column].isnull().sum()
        if null_before == 0:
            return df, PENALTY_WRONG_ACTION, \
                   f"No nulls in '{column}' — redundant action"

        df = df.copy()
        if strategy == "mean":
            numeric = pd.to_numeric(df[column], errors="coerce")
            df[column] = df[column].fillna(numeric.mean())
        elif strategy == "median":
            numeric = pd.to_numeric(df[column], errors="coerce")
            df[column] = df[column].fillna(numeric.median())
        elif strategy == "mode":
            df[column].fillna(df[column].mode()[0], inplace=True)
        elif strategy == "drop":
            before = len(df)
            df.dropna(subset=[column], inplace=True)
            dropped = before - len(df)
            if dropped > len(df) * 0.3:
                return df, PENALTY_DESTRUCTIVE, \
                       f"Dropped {dropped} rows — too destructive"
        else:
            df[column].fillna(strategy, inplace=True)

        null_after = df[column].isnull().sum()
        fixed      = null_before - null_after
        reward     = REWARD_ISSUE_FIXED * (fixed / max(null_before, 1))
        return df, reward, f"Filled {fixed} nulls in '{column}' via {strategy}"

    def _drop_duplicates(self, df: pd.DataFrame
                         ) -> Tuple[pd.DataFrame, float, str]:
        dup_before = df.duplicated().sum()
        if dup_before == 0:
            return df, PENALTY_WRONG_ACTION, "No duplicates — redundant action"

        df     = df.drop_duplicates().reset_index(drop=True)
        reward = REWARD_ISSUE_FIXED
        return df, reward, f"Removed {dup_before} duplicate rows"

    def _fix_dtype(self, df: pd.DataFrame,
                   params: Dict) -> Tuple[pd.DataFrame, float, str]:
        column      = params.get("column")
        target_type = params.get("target_type", "float")

        if column not in df.columns:
            return df, PENALTY_WRONG_ACTION, f"Column '{column}' not found"

        df = df.copy()
        try:
            if target_type == "int":
                df[column] = pd.to_numeric(
                    df[column], errors="coerce").fillna(0).astype(int)
            elif target_type == "float":
                df[column] = pd.to_numeric(df[column], errors="coerce")
            elif target_type == "str":
                df[column] = df[column].astype(str)
            elif target_type == "datetime":
                df[column] = pd.to_datetime(df[column], errors="coerce")
            else:
                return df, PENALTY_WRONG_ACTION, \
                       f"Unknown target type: {target_type}"
        except Exception as e:
            return df, PENALTY_WRONG_ACTION, f"Dtype fix failed: {e}"

        return df, REWARD_ISSUE_FIXED, \
               f"Converted '{column}' to {target_type}"

    def _rename_column(self, df: pd.DataFrame,
                       params: Dict) -> Tuple[pd.DataFrame, float, str]:
        old_name = params.get("old_name")
        new_name = params.get("new_name")

        if old_name not in df.columns:
            return df, PENALTY_WRONG_ACTION, \
                   f"Column '{old_name}' not found"

        expected = clean_column_name(old_name)
        if new_name != expected:
            return df, PENALTY_WRONG_ACTION, \
                   f"Suggested name '{new_name}' — expected '{expected}'"

        df = df.rename(columns={old_name: new_name})
        return df, REWARD_ISSUE_FIXED, \
               f"Renamed '{old_name}' to '{new_name}'"

    def _remove_outliers(self, df: pd.DataFrame,
                         params: Dict) -> Tuple[pd.DataFrame, float, str]:
        column = params.get("column")
        method = params.get("method", "iqr")

        if column not in df.columns:
            return df, PENALTY_WRONG_ACTION, f"Column '{column}' not found"

        df = df.copy()
        try:
            numeric = pd.to_numeric(df[column], errors="coerce")
            df[column] = numeric

            if method == "iqr":
                mask = detect_outliers_iqr(df, column)
            else:
                from utils import detect_outliers_zscore
                mask = detect_outliers_zscore(df, column)

            count = mask.sum()
            if count == 0:
                return df, PENALTY_WRONG_ACTION, \
                       f"No outliers in '{column}' — redundant"

            if count > len(df) * 0.4:
                return df, PENALTY_DESTRUCTIVE, \
                       f"Would remove {count} rows — too destructive"

            df = df[~mask].reset_index(drop=True)
            return df, REWARD_ISSUE_FIXED, \
                   f"Removed {count} outliers from '{column}'"

        except Exception as e:
            return df, PENALTY_WRONG_ACTION, f"Outlier removal failed: {e}"

    def _standardize_values(self, df: pd.DataFrame,
                             params: Dict
                             ) -> Tuple[pd.DataFrame, float, str]:
        column  = params.get("column")
        mapping = params.get("mapping", {})

        if column not in df.columns:
            return df, PENALTY_WRONG_ACTION, f"Column '{column}' not found"

        if not mapping:
            return df, PENALTY_WRONG_ACTION, "Empty mapping provided"

        before = df[column].value_counts().to_dict()
        df     = standardize_column(df, column, mapping)
        after  = df[column].value_counts().to_dict()

        changed = sum(1 for k, v in before.items()
                      if after.get(mapping.get(k, k), 0) != v)
        if changed == 0:
            return df, PENALTY_WRONG_ACTION, "No values changed"

        return df, REWARD_ISSUE_FIXED, \
               f"Standardized {len(mapping)} values in '{column}'"

    def _submit(self, df: pd.DataFrame
                ) -> Tuple[pd.DataFrame, float, str, Dict]:
        score, details = self.task.grade()
        self.done      = True

        if score >= 0.9:
            reward = REWARD_SUBMIT_BONUS + score
            reason = f"Excellent! Score: {score:.3f} — dataset is clean!"
        elif score >= 0.6:
            reward = score
            reason = f"Good effort. Score: {score:.3f} — some issues remain"
        else:
            reward = score * 0.5
            reason = f"Submitted early. Score: {score:.3f} — many issues remain"

        return df, reward, reason, {"final_score": score, "details": details}

    # ── Observation Builder ───────────────────

    def _make_observation(self) -> Observation:
        df = self.task.current_df
        return Observation(
            task_id        = self.task.task_id,
            difficulty     = self.difficulty,
            step           = self.step_count,
            max_steps      = MAX_STEPS,
            dataframe      = df_to_records(df),
            columns        = list(df.columns),
            dtypes         = {c: str(t) for c, t in df.dtypes.items()},
            null_counts    = get_null_counts(df),
            duplicate_rows = get_duplicate_count(df),
            issues         = self.task.issues,
            done           = self.done,
        )