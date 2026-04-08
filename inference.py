# inference.py
# ─────────────────────────────────────────────
# LLM agent runner — mandatory stdout format
# [START] [STEP] [END] as required by judges
# ─────────────────────────────────────────────

import os
import json
import time
from openai import OpenAI
from config import TASK_EASY, TASK_MEDIUM, TASK_HARD
from environment import DataCleaningEnv
from models import Action

# ── Client Setup ──────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "dummy-key")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ── System Prompt ─────────────────────────────
SYSTEM_PROMPT = """
You are a data cleaning agent. You will be given a messy CSV dataset
and must clean it step by step using the available actions.

Available actions:
1. fill_missing       — {"action_type": "fill_missing",       "parameters": {"column": "col_name", "strategy": "mean|median|mode|drop"}}
2. drop_duplicates    — {"action_type": "drop_duplicates",    "parameters": {}}
3. fix_dtype          — {"action_type": "fix_dtype",          "parameters": {"column": "col_name", "target_type": "int|float|str"}}
4. rename_column      — {"action_type": "rename_column",      "parameters": {"old_name": "Old Col", "new_name": "old_col"}}
5. remove_outliers    — {"action_type": "remove_outliers",    "parameters": {"column": "col_name", "method": "iqr|zscore"}}
6. standardize_values — {"action_type": "standardize_values", "parameters": {"column": "col_name", "mapping": {"old": "new"}}}
7. submit             — {"action_type": "submit",             "parameters": {}}

Rules:
- Always respond with a single valid JSON action only
- No explanation, no markdown, just raw JSON
- Call submit when you think the dataset is clean
- Analyze the issues list carefully before acting
"""

# ── Agent Loop ────────────────────────────────
def run_agent(difficulty: str, max_steps: int = 20) -> dict:
    env      = DataCleaningEnv(difficulty=difficulty)
    obs      = env.reset()
    history  = []
    rewards  = []
    success  = False
    score    = 0.0

    task_name = f"data-cleaning-{difficulty}"

    # ── [START] ──
    print(f"[START] task={task_name} env=data-cleaning-env model={MODEL_NAME}")

    for step in range(1, max_steps + 1):
        if obs.done:
            break

        user_msg = f"""
Current state:
- Task: {obs.task_id}
- Step: {obs.step}/{obs.max_steps}
- Issues: {obs.issues}
- Null counts: {obs.null_counts}
- Duplicate rows: {obs.duplicate_rows}
- Columns: {obs.columns}
- Dtypes: {obs.dtypes}
- Data (first 5 rows): {obs.dataframe[:5]}

Respond with JSON action only.
"""
        history.append({"role": "user", "content": user_msg})

        # Call LLM
        error_msg = "null"
        action_str = "null"
        reward_val = 0.0
        done_val   = False

        try:
            response = client.chat.completions.create(
                model    = MODEL_NAME,
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *history,
                ],
                max_tokens  = 200,
                temperature = 0.0,
            )
            raw = response.choices[0].message.content.strip()
            history.append({"role": "assistant", "content": raw})

            # Parse action
            action_dict = json.loads(raw)
            action      = Action(**action_dict)
            action_str  = f"{action.action_type}({json.dumps(action.parameters)})"

            # Execute
            result     = env.step(action)
            reward_val = result.reward.value
            done_val   = result.done
            obs        = result.observation

            if result.reward.reason and "failed" in result.reward.reason.lower():
                error_msg = result.reward.reason[:80]

        except json.JSONDecodeError as e:
            error_msg  = f"json_parse_error"
            done_val   = False
        except Exception as e:
            error_msg  = str(e)[:80].replace("\n", " ")
            done_val   = False

        rewards.append(reward_val)

        # ── [STEP] ──
        done_str = "true" if done_val else "false"
        print(
            f"[STEP] step={step} action={action_str} "
            f"reward={reward_val:.2f} done={done_str} error={error_msg}"
        )

        if done_val:
            break

        time.sleep(0.3)

    # Final score
    score, _ = env.task.grade()
    success  = score >= 0.6

    rewards_str  = ",".join(f"{r:.2f}" for r in rewards)
    success_str  = "true" if success else "false"
    steps_taken  = len(rewards)

    # ── [END] ──
    print(f"[END] success={success_str} steps={steps_taken} score={score:.2f} rewards={rewards_str}")


    return {
        "difficulty":  difficulty,
        "steps_taken": steps_taken,
        "score":       score,
        "success":     success,
        "rewards":     rewards,
    }

# ── Main ──────────────────────────────────────
def main():
    results = {}
    for difficulty in [TASK_EASY, TASK_MEDIUM, TASK_HARD]:
        results[difficulty] = run_agent(difficulty)
        time.sleep(1)
    return results

if __name__ == "__main__":
    main()