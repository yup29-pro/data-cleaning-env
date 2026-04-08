---
title: Data Cleaning Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# Data Cleaning Environment — OpenEnv

An OpenEnv-compliant environment where an AI agent cleans messy CSV datasets step by step.

## Environment Description

The agent receives a messy CSV dataset and must issue cleaning actions to fix issues like missing values, duplicates, outliers, bad column names, and inconsistent values.

## Action Space

| Action | Parameters | Description |
|---|---|---|
| `fill_missing` | column, strategy (mean/median/mode/drop) | Fill null values |
| `drop_duplicates` | — | Remove duplicate rows |
| `fix_dtype` | column, target_type (int/float/str) | Convert column dtype |
| `rename_column` | old_name, new_name | Rename to snake_case |
| `remove_outliers` | column, method (iqr/zscore) | Remove outlier rows |
| `standardize_values` | column, mapping (JSON) | Unify inconsistent values |
| `submit` | — | Finalize and trigger grader |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Current task identifier |
| `difficulty` | string | easy / medium / hard |
| `step` | int | Current step number |
| `max_steps` | int | Maximum steps allowed |
| `dataframe` | array | Current dataset as list of dicts |
| `columns` | array | Column names |
| `dtypes` | object | Column dtype mapping |
| `null_counts` | object | Null count per column |
| `duplicate_rows` | int | Number of duplicate rows |
| `issues` | array | Detected issues list |
| `done` | bool | Episode over? |

## Tasks

### Task 1 — Easy (score: 0.0–1.0)
Fix a 10-row employee dataset:
- Fill missing values in `age` and `salary`
- Convert `age` from string to integer

Baseline score: **0.72**

### Task 2 — Medium (score: 0.0–1.0)
Fix a 20-row dataset:
- Remove duplicate rows
- Handle salary outliers (IQR method)
- Standardize country values → `United States` or `UK`

Baseline score: **0.65**

### Task 3 — Hard (score: 0.0–1.0)
Fix a 30-row dataset with all issues:
- Fix bad column names (trailing spaces, uppercase, special chars)
- Fill missing values, remove duplicates and outliers
- Standardize country values
- Fix invalid `dept_id` values (referential integrity)

Baseline score: **0.48**

## Reward Function

| Event | Reward |
|---|---|
| Issue fixed | +0.10 |
| Clean submit bonus | +0.30 |
| Wrong/redundant action | -0.05 |
| Destructive action | -0.10 |

## Setup Instructions

### Local
```bash
git clone https://huggingface.co/spaces/Yashwanth34567/data-cleaning-env
cd data-cleaning-env
pip install -r requirements.txt
python app.py
```

### Docker
```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

### Environment Variables
```bash
API_BASE_URL=https://api-inference.huggingface.co/v1
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
OPENAI_API_KEY=your_hf_token
HF_TOKEN=your_hf_token
```

## Run Inference
```bash
python inference.py
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start fresh episode |
| `/step` | POST | Execute action |
| `/state` | GET | Current environment state |
| `/tasks` | GET | List all tasks |
| `/openenv.yaml` | GET | OpenEnv spec |



## Live Demo

🚀 **App UI:** https://yashwanth34567-data-cleaning-env.hf.space/ui

🔗 **API Reset:** https://yashwanth34567-data-cleaning-env.hf.space/reset

🔗 **API State:** https://yashwanth34567-data-cleaning-env.hf.space/state

🔗 **API Health:** https://yashwanth34567-data-cleaning-env.hf.space/health

## Author

**Yashwanth R (Yashwanth34567)** & **Mohammed Ayaan** — OpenEnv Hackathon × Scaler School of Technology