# models.py
# ─────────────────────────────────────────────
# Pydantic models for OpenEnv spec compliance
# Observation, Action, Reward typed models
# ─────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

# ── Observation ───────────────────────────────
class Observation(BaseModel):
    """What the agent sees at each step"""

    task_id:         str                    = Field(..., description="Current task identifier")
    difficulty:      str                    = Field(..., description="easy / medium / hard")
    step:            int                    = Field(..., description="Current step number")
    max_steps:       int                    = Field(..., description="Maximum steps allowed")
    dataframe:       List[Dict[str, Any]]   = Field(..., description="Current data as list of row dicts")
    columns:         List[str]              = Field(..., description="Column names")
    dtypes:          Dict[str, str]         = Field(..., description="Column -> dtype mapping")
    null_counts:     Dict[str, int]         = Field(..., description="Column -> null count")
    duplicate_rows:  int                    = Field(..., description="Number of duplicate rows")
    issues:          List[str]              = Field(..., description="List of detected issues")
    done:            bool                   = Field(False, description="Is episode over?")

# ── Action ────────────────────────────────────
class Action(BaseModel):
    """What the agent can do"""

    action_type: str                        = Field(..., description="""
        One of:
        fill_missing | drop_duplicates | fix_dtype |
        rename_column | remove_outliers |
        standardize_values | submit
    """)
    parameters:  Dict[str, Any]             = Field(default_factory=dict, description="Action parameters")

# ── Reward ────────────────────────────────────
class Reward(BaseModel):
    """Score returned after each step"""

    value:    float  = Field(..., description="Reward value for this step")
    total:    float  = Field(..., description="Cumulative reward so far")
    reason:   str    = Field(..., description="Why this reward was given")

# ── Step Result ───────────────────────────────
class StepResult(BaseModel):
    """Full return value of step()"""

    observation: Observation
    reward:      Reward
    done:        bool
    info:        Dict[str, Any] = Field(default_factory=dict)

# ── Task Info ─────────────────────────────────
class TaskInfo(BaseModel):
    """Metadata about a task"""

    task_id:     str
    difficulty:  str
    description: str
    max_steps:   int
    score:       Optional[float] = None