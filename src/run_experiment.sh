#!/bin/bash
# Experiment runner with GitHub PR tracking
# Creates a branch, runs the task, commits artifacts at milestones, creates/updates a PR.
# The agent is instructed to commit its work — a background sync pushes commits to GitHub.
#
# Usage: bash src/run_experiment.sh <eval> <agent> <model_to_train> <num_hours> <agent_config> [gpu_id]
#
# Example:
#   bash src/run_experiment.sh gsm8k claude Qwen/Qwen3-1.7B-Base 10 claude-opus-4-6 0

set -o pipefail

if [ $# -lt 5 ]; then
    echo "Usage: $0 <eval> <agent> <model_to_train> <num_hours> <agent_config> [gpu_id]" >&2
    exit 1
fi

EVALUATION_TASK="$1"
AGENT="$2"
MODEL_TO_TRAIN="$3"
NUM_HOURS="$4"
AGENT_CONFIG="$5"
GPU_ID="${6:-0}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

gh repo set-default "$(git remote get-url origin | sed 's|.*github.com/||;s|\.git$||')" 2>/dev/null || true

# --- Naming ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SHORT=$(echo "$MODEL_TO_TRAIN" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
CONFIG_SHORT=$(echo "$AGENT_CONFIG" | tr '/:' '-')
RUN_NAME="${EVALUATION_TASK}_${MODEL_SHORT}_${CONFIG_SHORT}_${TIMESTAMP}"
BRANCH_NAME="exp/${RUN_NAME}"
EXP_DIR="experiments/${RUN_NAME}"
RUN_ID="${TIMESTAMP}"

echo "=== PostTrainBench Experiment Runner ==="
echo "Run: ${RUN_NAME}"
echo "Branch: ${BRANCH_NAME}"
echo ""

# --- Ensure we're on main and up to date ---
git checkout main 2>/dev/null
git pull origin main --ff-only 2>/dev/null || true

# --- Create branch from main ---
git checkout -b "$BRANCH_NAME" main 2>/dev/null || { echo "ERROR: Failed to create branch"; exit 1; }

# --- Override results/.gitignore so agent commits are tracked ---
echo '!*' > results/.gitignore.exp
cat > results/.gitignore.exp <<'GITIGNORE'
# Allow tracking experiment files (overrides the default * ignore)
!*/
!*.py
!*.sh
!*.json
!*.txt
!*.csv
!*.jinja
!*.log
!*.md
!*.yaml
!*.yml
!*.toml
# Still ignore model weights and caches
*.pt
*.bin
*.safetensors
*.gguf
__pycache__/
.cache/
wandb/
GITIGNORE
cp results/.gitignore.exp results/.gitignore

# --- Create experiment directory ---
mkdir -p "$EXP_DIR"

GPU_NAME="$(nvidia-smi --id=$GPU_ID --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
cat > "${EXP_DIR}/config.json" <<CONF
{
  "evaluation_task": "${EVALUATION_TASK}",
  "agent": "${AGENT}",
  "agent_config": "${AGENT_CONFIG}",
  "model_to_train": "${MODEL_TO_TRAIN}",
  "num_hours": ${NUM_HOURS},
  "gpu_id": ${GPU_ID},
  "run_name": "${RUN_NAME}",
  "timestamp": "${TIMESTAMP}",
  "hostname": "$(hostname)",
  "gpu_name": "${GPU_NAME}"
}
CONF

# --- Commit config and push ---
git add results/.gitignore "${EXP_DIR}/config.json"
git commit -m "Start experiment: ${EVALUATION_TASK} / ${MODEL_SHORT} / ${CONFIG_SHORT}"
git push -u origin "$BRANCH_NAME" 2>&1

sleep 3

# --- Create PR ---
PR_BODY="## Experiment: ${EVALUATION_TASK} / ${MODEL_SHORT} / ${CONFIG_SHORT}

### Config
| Parameter | Value |
|-----------|-------|
| Benchmark | \`${EVALUATION_TASK}\` |
| Model | \`${MODEL_TO_TRAIN}\` |
| Agent | \`${AGENT}\` (\`${AGENT_CONFIG}\`) |
| GPU | ${GPU_ID} (${GPU_NAME}) |
| Time limit | ${NUM_HOURS}h |
| Started | $(date -u '+%Y-%m-%d %H:%M UTC') |

### Status
:hourglass_flowing_sand: **Running** — agent is working. Check commits for live progress.

### Results
_Pending..._

### Artifacts
- [\`config.json\`](experiments/${RUN_NAME}/config.json) — Run configuration
- Check the **Commits** tab for live agent progress"

PR_URL=$(gh pr create \
    --title "[Exp] ${EVALUATION_TASK} / ${MODEL_SHORT} / ${CONFIG_SHORT}" \
    --body "$PR_BODY" \
    --draft \
    --head "$BRANCH_NAME" \
    --base main 2>&1) || true

if [[ "$PR_URL" == http* ]]; then
    echo "PR created: ${PR_URL}"
    echo "$PR_URL" > "${EXP_DIR}/.pr_url"
else
    echo "WARNING: Failed to create PR: ${PR_URL}" >&2
    echo "" > "${EXP_DIR}/.pr_url"
fi

# --- Background sync: push agent's commits to GitHub every 2 minutes ---
SYNC_PID_FILE="/tmp/ptb_sync_${RUN_ID}.pid"
(
    while true; do
        sleep 120
        # Check if the experiment is still running
        if ! ps -p $$ > /dev/null 2>&1; then
            break
        fi
        # Push any new commits
        cd "$REPO_ROOT"
        if git log origin/${BRANCH_NAME}..HEAD --oneline 2>/dev/null | grep -q .; then
            git push origin "$BRANCH_NAME" 2>/dev/null && echo "[sync] Pushed commits at $(date '+%H:%M:%S')"
        fi
    done
) &
SYNC_PID=$!
echo "$SYNC_PID" > "$SYNC_PID_FILE"
echo "Background sync started (PID: $SYNC_PID)"

# --- Run the actual task ---
echo ""
echo "=== Starting task execution ==="

export POST_TRAIN_BENCH_JOB_SCHEDULER="${POST_TRAIN_BENCH_JOB_SCHEDULER:-none}"
export POST_TRAIN_BENCH_RESULTS_DIR="${POST_TRAIN_BENCH_RESULTS_DIR:-results}"
export POST_TRAIN_BENCH_EXPERIMENT_NAME="${POST_TRAIN_BENCH_EXPERIMENT_NAME:-}"
export POST_TRAIN_BENCH_PROMPT="${POST_TRAIN_BENCH_PROMPT:-prompt_native}"

bash src/run_task_native.sh \
    "$EVALUATION_TASK" \
    "$AGENT" \
    "$MODEL_TO_TRAIN" \
    "$RUN_ID" \
    "$NUM_HOURS" \
    "$AGENT_CONFIG" \
    "$GPU_ID"

TASK_EXIT=$?

# --- Stop background sync ---
kill "$SYNC_PID" 2>/dev/null
rm -f "$SYNC_PID_FILE"

# --- Locate results ---
RESULT_PREFIX_SAFE=$(echo "$MODEL_TO_TRAIN" | tr '/:' '_')
AGENT_CONFIG_SAFE=$(echo "$AGENT_CONFIG" | tr '/:' '_')
EVAL_DIR="${REPO_ROOT}/${POST_TRAIN_BENCH_RESULTS_DIR}/${AGENT}_${AGENT_CONFIG_SAFE}_${NUM_HOURS}h${POST_TRAIN_BENCH_EXPERIMENT_NAME}/${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${RUN_ID}"

echo ""
echo "=== Committing final experiment artifacts ==="

# --- Copy artifacts to experiment dir ---
for f in prompt.txt solve_parsed.txt time_taken.txt metrics.json; do
    if [ -f "${EVAL_DIR}/${f}" ]; then
        cp "${EVAL_DIR}/${f}" "${EXP_DIR}/${f}"
    fi
done

for f in output error; do
    if [ -f "${EVAL_DIR}/${f}.log" ]; then
        cp "${EVAL_DIR}/${f}.log" "${EXP_DIR}/${f}_log.txt"
    fi
done

# --- Determine status ---
if [ -f "${EXP_DIR}/metrics.json" ]; then
    STATUS="completed"
    STATUS_EMOJI=":white_check_mark:"
    STATUS_TEXT="Completed"
    METRICS_TABLE=$(python3 -c "
import json
with open('${EXP_DIR}/metrics.json') as f:
    m = json.load(f)
for k, v in m.items():
    if isinstance(v, float):
        print(f'| {k} | {v:.4f} |')
    else:
        print(f'| {k} | {v} |')
" 2>/dev/null || echo "| error | could not parse metrics |")
elif [ $TASK_EXIT -eq 124 ]; then
    STATUS="timeout"
    STATUS_EMOJI=":alarm_clock:"
    STATUS_TEXT="Timed out (${NUM_HOURS}h limit)"
    METRICS_TABLE="| N/A | task timed out |"
else
    STATUS="failed"
    STATUS_EMOJI=":x:"
    STATUS_TEXT="Failed (exit code: ${TASK_EXIT})"
    METRICS_TABLE="| N/A | task failed |"
fi

TIME_TAKEN="unknown"
if [ -f "${EXP_DIR}/time_taken.txt" ]; then
    TIME_TAKEN=$(cat "${EXP_DIR}/time_taken.txt")
fi

cat > "${EXP_DIR}/status.json" <<STAT
{
  "status": "${STATUS}",
  "exit_code": ${TASK_EXIT},
  "time_taken": "${TIME_TAKEN}",
  "completed_at": "$(date -u '+%Y-%m-%d %H:%M UTC')"
}
STAT

# --- Final commit and push ---
git add -A
git commit -m "Final: ${EVALUATION_TASK} / ${MODEL_SHORT} — ${STATUS} (${TIME_TAKEN})" || true
git push origin "$BRANCH_NAME" 2>&1

# --- Update PR ---
UPDATED_BODY="## Experiment: ${EVALUATION_TASK} / ${MODEL_SHORT} / ${CONFIG_SHORT}

### Config
| Parameter | Value |
|-----------|-------|
| Benchmark | \`${EVALUATION_TASK}\` |
| Model | \`${MODEL_TO_TRAIN}\` |
| Agent | \`${AGENT}\` (\`${AGENT_CONFIG}\`) |
| GPU | ${GPU_ID} (${GPU_NAME}) |
| Time limit | ${NUM_HOURS}h |
| Duration | ${TIME_TAKEN} |

### Status
${STATUS_EMOJI} **${STATUS_TEXT}**

### Results
| Metric | Value |
|--------|-------|
${METRICS_TABLE}

### How to review
- **Commits tab**: See every step the agent took (training scripts, config changes, pivots)
- **Files changed**: See all code the agent wrote
- \`experiments/${RUN_NAME}/solve_parsed.txt\`: Full agent decision trajectory"

gh pr edit "$BRANCH_NAME" --body "$UPDATED_BODY" 2>&1 || true

if [ "$STATUS" = "completed" ]; then
    gh pr ready "$BRANCH_NAME" 2>&1 || true
fi

echo ""
echo "=== Experiment complete ==="
echo "Status: ${STATUS_TEXT}"
echo "Time: ${TIME_TAKEN}"
echo "PR: $(cat ${EXP_DIR}/.pr_url 2>/dev/null || echo 'unknown')"
echo "Results: ${EXP_DIR}/"

git checkout main 2>/dev/null
