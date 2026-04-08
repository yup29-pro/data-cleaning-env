# config.py
# ─────────────────────────────────────────────
# Central configuration for Data Cleaning Env
# ─────────────────────────────────────────────

import os

# ── Environment Metadata ──────────────────────
ENV_NAME        = "data-cleaning-env"
ENV_VERSION     = "1.0.0"
ENV_DESCRIPTION = "An OpenEnv environment where an AI agent cleans messy CSV datasets"
AUTHOR          = "Yashwanth34567"

# ── Task Difficulty Levels ────────────────────
TASK_EASY   = "easy"
TASK_MEDIUM = "medium"
TASK_HARD   = "hard"

TASK_IDS = {
    TASK_EASY:   "task_1_easy",
    TASK_MEDIUM: "task_2_medium",
    TASK_HARD:   "task_3_hard",
}

# ── Reward Values ─────────────────────────────
REWARD_ISSUE_FIXED       =  0.10   # each real issue fixed
REWARD_SUBMIT_BONUS      =  0.30   # bonus for clean submit
PENALTY_WRONG_ACTION     = -0.05   # redundant / wrong action
PENALTY_DESTRUCTIVE      = -0.10   # dropping valid data

# ── Episode Settings ──────────────────────────
MAX_STEPS = 20   # max actions per episode

# ── API Settings ──────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

# ── Server Settings ───────────────────────────
HOST = "0.0.0.0"
PORT = 8000