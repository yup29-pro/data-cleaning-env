# app.py
# ─────────────────────────────────────────────
# Gradio UI for Data Cleaning Environment
# Beautiful dashboard with Playfair Display
# ─────────────────────────────────────────────

import gradio as gr
import pandas as pd
import json
from environment import DataCleaningEnv
from models import Action
from config import TASK_EASY, TASK_MEDIUM, TASK_HARD
import threading
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# ── Global State ──────────────────────────────
envs = {
    TASK_EASY:   DataCleaningEnv(TASK_EASY),
    TASK_MEDIUM: DataCleaningEnv(TASK_MEDIUM),
    TASK_HARD:   DataCleaningEnv(TASK_HARD),
}
current_difficulty = TASK_EASY
step_logs = {TASK_EASY: [], TASK_MEDIUM: [], TASK_HARD: []}

# ── Helpers ───────────────────────────────────

def get_env():
    return envs[current_difficulty]

def df_to_html(df: pd.DataFrame, null_counts: dict) -> str:
    """Render DataFrame as styled HTML table"""
    if df is None or len(df) == 0:
        return "<p style='color:#64748B;font-size:13px'>No data</p>"

    rows_html = ""
    for _, row in df.iterrows():
        cells = ""
        for col in df.columns:
            val = row[col]
            is_null = val is None or (
                isinstance(val, float) and pd.isna(val)
            )
            if is_null:
                cells += f"<td style='color:#EF4444;font-style:italic;background:#FFF5F5;padding:7px 12px;border-bottom:1px solid #F1F5F9'>null</td>"
            elif isinstance(val, (int, float)) and col in ["salary", "salary$"]:
                if abs(float(val)) > 500000 or float(val) < 0:
                    cells += f"<td style='color:#D97706;background:#FFFBEB;padding:7px 12px;border-bottom:1px solid #F1F5F9'>{val}</td>"
                else:
                    cells += f"<td style='padding:7px 12px;border-bottom:1px solid #F1F5F9;color:#1E293B'>{val}</td>"
            else:
                cells += f"<td style='padding:7px 12px;border-bottom:1px solid #F1F5F9;color:#1E293B'>{val}</td>"
        rows_html += f"<tr>{cells}</tr>"

    headers = "".join([
        f"<th style='background:#F8FAFC;padding:8px 12px;text-align:left;font-size:11px;color:#64748B;font-weight:500;border-bottom:1px solid #E2E8F0'>{col}</th>"
        for col in df.columns
    ])

    return f"""
    <div style='background:#fff;border:1px solid #E2E8F0;border-radius:10px;overflow:hidden;font-family:DM Sans,sans-serif'>
      <div style='padding:10px 14px;border-bottom:1px solid #E2E8F0;display:flex;justify-content:space-between;align-items:center'>
        <span style='font-size:13px;font-weight:500;color:#1E293B'>Live dataset</span>
        <span style='background:#EFF6FF;color:#2563EB;font-size:11px;padding:3px 10px;border-radius:20px;border:1px solid #BFDBFE'>{len(df)} rows · {len(df.columns)} cols</span>
      </div>
      <div style='overflow-x:auto'>
        <table style='width:100%;border-collapse:collapse;font-size:12px;color:#1E293B'>
          <thead><tr>{headers}</tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </div>
    """

def issues_to_html(issues: list) -> str:
    """Render issues as styled pills"""
    if not issues:
        return "<div style='color:#10B981;font-size:13px;font-weight:500'>No issues detected!</div>"

    pills = ""
    for issue in issues:
        if "missing" in issue:
            color = "#FFF5F5"; text = "#B91C1C"; border = "#FECACA"
        elif "duplicate" in issue:
            color = "#FFFBEB"; text = "#92400E"; border = "#FDE68A"
        elif "outlier" in issue:
            color = "#FFFBEB"; text = "#92400E"; border = "#FDE68A"
        elif "bad_column" in issue:
            color = "#EFF6FF"; text = "#1D4ED8"; border = "#BFDBFE"
        else:
            color = "#F8FAFC"; text = "#475569"; border = "#E2E8F0"

        pills += f"<div style='background:{color};color:{text};border:1px solid {border};font-size:11px;padding:5px 10px;border-radius:6px;margin-bottom:6px'>{issue}</div>"

    return pills

def log_to_html(logs: list) -> str:
    """Render step log as terminal-style HTML"""
    if not logs:
        return "<div style='background:#0F172A;border-radius:8px;padding:12px;font-size:11px;font-family:monospace;color:#475569'>No steps yet...</div>"

    lines = ""
    for entry in logs[-10:]:
        reward_color = "#34D399" if entry["reward"] > 0 else "#F87171"
        lines += f"""
        <div style='margin-bottom:6px'>
          <span style='color:#60A5FA'>[step {entry['step']}]</span>
          <span style='color:#E2E8F0'> {entry['action']}</span>
          <br>
          <span style='color:{reward_color};padding-left:12px'>{entry['reward']:+.3f} · {entry['reason'][:50]}</span>
        </div>"""

    return f"<div style='background:#0F172A;border-radius:8px;padding:12px;font-size:11px;font-family:monospace;line-height:1.7;max-height:220px;overflow-y:auto'>{lines}</div>"


# ── Actions ───────────────────────────────────

def reset_task(difficulty):
    global current_difficulty
    current_difficulty = difficulty
    env = envs[difficulty]
    obs = env.reset()
    step_logs[difficulty] = []

    df = pd.DataFrame(obs.dataframe)
    score, _ = env.task.grade()

    return (
        df_to_html(df, obs.null_counts),
        issues_to_html(obs.issues),
        log_to_html([]),
        f"<div style='font-family:Playfair Display,serif;font-size:28px;font-weight:700;color:#2563EB'>0.000</div>",
        f"<div style='font-family:Playfair Display,serif;font-size:28px;font-weight:700;color:#1E293B'>0</div>",
        f"<div style='font-family:Playfair Display,serif;font-size:28px;font-weight:700;color:#EF4444'>{sum(obs.null_counts.values())}</div>",
        f"<div style='font-family:Playfair Display,serif;font-size:28px;font-weight:700;color:#F59E0B'>0</div>",
    )


def run_action(difficulty, action_type, column, strategy,
               target_type, old_name, new_name, mapping_json):
    global current_difficulty
    current_difficulty = difficulty
    env = envs[difficulty]

    if not env._initialized:
        env.reset()

    # Build parameters
    params = {}
    if action_type == "fill_missing":
        params = {"column": column, "strategy": strategy or "mean"}
    elif action_type == "drop_duplicates":
        params = {}
    elif action_type == "fix_dtype":
        params = {"column": column, "target_type": target_type or "float"}
    elif action_type == "rename_column":
        params = {"old_name": old_name, "new_name": new_name}
    elif action_type == "remove_outliers":
        params = {"column": column, "method": strategy or "iqr"}
    elif action_type == "standardize_values":
        try:
            params = {"column": column, "mapping": json.loads(mapping_json)}
        except Exception:
            params = {"column": column, "mapping": {}}
    elif action_type == "submit":
        params = {}

    action = Action(action_type=action_type, parameters=params)
    result = env.step(action)
    obs    = result.observation

    # Log step
    step_logs[difficulty].append({
        "step":   env.step_count,
        "action": action_type,
        "reward": result.reward.value,
        "reason": result.reward.reason,
    })

    df    = pd.DataFrame(obs.dataframe)
    score, _ = env.task.grade()

    return (
        df_to_html(df, obs.null_counts),
        issues_to_html(obs.issues),
        log_to_html(step_logs[difficulty]),
        f"<div style='font-family:Playfair Display,serif;font-size:28px;font-weight:700;color:#2563EB'>{score:.3f}</div>",
        f"<div style='font-family:Playfair Display,serif;font-size:28px;font-weight:700;color:#1E293B'>{env.step_count}</div>",
        f"<div style='font-family:Playfair Display,serif;font-size:28px;font-weight:700;color:#EF4444'>{sum(obs.null_counts.values())}</div>",
        f"<div style='font-family:Playfair Display,serif;font-size:28px;font-weight:700;color:{'#10B981' if result.reward.value > 0 else '#EF4444'}'>{result.reward.value:+.3f}</div>",
    )


# ── UI ────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');
body, .gradio-container { background: #F8FAFC !important; font-family: 'DM Sans', sans-serif !important; }
.logo { font-family: 'Playfair Display', serif !important; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; }
.gr-button-primary { background: #2563EB !important; border: none !important; }
.gr-button { border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important; }
"""

with gr.Blocks(css=CSS, title="DataClean — OpenEnv") as demo:

    gr.HTML("""
    <div style='background:#fff;border-bottom:1px solid #E2E8F0;padding:16px 24px;display:flex;align-items:center;justify-content:space-between;margin-bottom:0'>
      <div style='font-family:Playfair Display,serif;font-size:26px;font-weight:900;color:#2563EB;letter-spacing:-0.5px'>
        Data<span style='color:#1E293B'>Clean</span>
      </div>
      <div style='background:#EFF6FF;color:#2563EB;font-size:12px;font-weight:500;padding:4px 14px;border-radius:20px;border:1px solid #BFDBFE'>
        OpenEnv v1.0 · Yashwanth34567
      </div>
    </div>
    """)

    with gr.Row():

        # ── Left Sidebar ──
        with gr.Column(scale=1):
            gr.HTML("<div style='font-size:11px;font-weight:500;color:#64748B;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px'>Select Task</div>")
            difficulty = gr.Radio(
                choices=[TASK_EASY, TASK_MEDIUM, TASK_HARD],
                value=TASK_EASY,
                label="",
            )
            reset_btn = gr.Button("Reset Task", variant="primary")

            gr.HTML("<div style='font-size:11px;font-weight:500;color:#64748B;letter-spacing:0.08em;text-transform:uppercase;margin:16px 0 8px'>Issues</div>")
            issues_display = gr.HTML()

        # ── Center ──
        with gr.Column(scale=3):
            with gr.Row():
                score_display  = gr.HTML("<div style='font-family:Playfair Display,serif;font-size:28px;font-weight:700;color:#2563EB'>—</div>")
                step_display   = gr.HTML("<div style='font-family:Playfair Display,serif;font-size:28px;font-weight:700;color:#1E293B'>—</div>")
                null_display   = gr.HTML("<div style='font-family:Playfair Display,serif;font-size:28px;font-weight:700;color:#EF4444'>—</div>")
                reward_display = gr.HTML("<div style='font-family:Playfair Display,serif;font-size:28px;font-weight:700;color:#F59E0B'>—</div>")

            gr.HTML("<div style='font-size:11px;font-weight:500;color:#64748B;letter-spacing:0.08em;text-transform:uppercase;margin:8px 0'>Live Dataset</div>")
            table_display = gr.HTML()

            gr.HTML("<div style='font-size:11px;font-weight:500;color:#64748B;letter-spacing:0.08em;text-transform:uppercase;margin:16px 0 8px'>Step Log</div>")
            log_display = gr.HTML()

        # ── Right Panel ──
        with gr.Column(scale=1):
            gr.HTML("<div style='font-size:11px;font-weight:500;color:#64748B;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px'>Take Action</div>")

            action_type = gr.Dropdown(
                choices=["fill_missing", "drop_duplicates", "fix_dtype",
                         "rename_column", "remove_outliers",
                         "standardize_values", "submit"],
                value="fill_missing",
                label="Action",
            )
            column      = gr.Textbox(label="Column", placeholder="e.g. age")
            strategy    = gr.Textbox(label="Strategy / Method", placeholder="mean | median | mode | iqr")
            target_type = gr.Textbox(label="Target Type", placeholder="int | float | str")
            old_name    = gr.Textbox(label="Old Column Name", placeholder="Full Name ")
            new_name    = gr.Textbox(label="New Column Name", placeholder="full_name")
            mapping     = gr.Textbox(label='Mapping (JSON)', placeholder='{"USA": "United States"}')

            run_btn    = gr.Button("Run Action", variant="primary")
            submit_btn = gr.Button("Submit Task", variant="secondary")

    # ── Events ───────────────────────────────
    outputs = [
        table_display, issues_display, log_display,
        score_display, step_display, null_display, reward_display
    ]

    reset_btn.click(
        fn=reset_task,
        inputs=[difficulty],
        outputs=outputs,
    )

    run_btn.click(
        fn=run_action,
        inputs=[difficulty, action_type, column, strategy,
                target_type, old_name, new_name, mapping],
        outputs=outputs,
    )

    submit_btn.click(
        fn=lambda d, c, s, t, o, n, m: run_action(
            d, "submit", c, s, t, o, n, m),
        inputs=[difficulty, column, strategy,
                target_type, old_name, new_name, mapping],
        outputs=outputs,
    )
    
# ── FastAPI ───────────────────────────────────
api     = FastAPI()
api_env = DataCleaningEnv(TASK_EASY)

@api.post("/reset")
def api_reset():
    obs = api_env.reset()
    return JSONResponse(content=json.loads(obs.model_dump_json()))

@api.post("/step")
def api_step(action: Action):
    result = api_env.step(action)
    return JSONResponse(content=json.loads(result.model_dump_json()))

@api.get("/state")
def api_state():
    try:
        state = api_env.state()
        return JSONResponse(content=json.loads(json.dumps(state, default=str)))
    except Exception as e:
        # Force initialize if not started
        api_env.reset()
        state = api_env.state()
        return JSONResponse(content=json.loads(json.dumps(state, default=str)))

@api.post("/step")
def api_step(action: Action):
    result = api_env.step(action)
    return JSONResponse(content=result.model_dump())

@api.get("/state")
def api_state():
    return JSONResponse(content=api_env.state())

@api.get("/health")
def health():
    return {"status": "ok"}
from fastapi.responses import RedirectResponse

@api.get("/")
def root():
    return RedirectResponse(url="/ui")

if __name__ == "__main__":
    app_with_api = gr.mount_gradio_app(api, demo, path="/ui")
    uvicorn.run(app_with_api, host="0.0.0.0", port=7860)