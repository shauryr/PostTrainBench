#!/bin/bash
# Experiment runner with GitHub PR tracking
# Creates a branch, runs the task, commits artifacts at milestones, creates/updates a PR.
#
# Usage: bash src/run_experiment.sh <eval> <agent> <model_to_train> <num_hours> <agent_config> [gpu_id]
#
# Example:
#   bash src/run_experiment.sh gsm8k claude Qwen/Qwen3-1.7B-Base 10 claude-opus-4-6 0
#
# Each run creates:
#   - A git branch: exp/<eval>_<model-short>_<agent-config-short>_<timestamp>
#   - A GitHub PR with status updates and metrics
#   - Artifacts under experiments/<run-name>/

set -o pipefail

EVALUATION_TASK="$1"
AGENT="$2"
MODEL_TO_TRAIN="$3"
NUM_HOURS="$4"
AGENT_CONFIG="$5"
GPU_ID="${6:-0}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Ensure gh knows which repo to target
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

# --- Create experiment directory ---
mkdir -p "$EXP_DIR"

# --- Write config ---
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
git add "${EXP_DIR}/config.json"
git commit -m "Start experiment: ${EVALUATION_TASK} / ${MODEL_SHORT} / ${CONFIG_SHORT}"
git push -u origin "$BRANCH_NAME" 2>&1

# Wait for GitHub to register the branch
sleep 3

# --- Create PR ---
PR_BODY="$(cat <<'PRBODYEOF'
## Experiment: EVAL_TASK / MODEL / CONFIG

### Config
| Parameter | Value |
|-----------|-------|
PRBODYEOF
)"
# Build PR body with actual values
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
:hourglass_flowing_sand: **Running** — agent is working...

### Results
_Pending..._

### Artifacts
- [\`config.json\`](experiments/${RUN_NAME}/config.json) — Run configuration"

PR_URL=$(gh pr create \
    --title "[Exp] ${EVALUATION_TASK} / ${MODEL_SHORT} / ${CONFIG_SHORT}" \
    --body "$PR_BODY" \
    --draft \
    --head "$BRANCH_NAME" \
    --base main 2>&1) || true

echo "PR: ${PR_URL}"
echo "$PR_URL" > "${EXP_DIR}/.pr_url"

# --- Run the actual task ---
echo ""
echo "=== Starting task execution ==="

# Pre-set env vars needed by run_task_native.sh
export POST_TRAIN_BENCH_JOB_SCHEDULER="${POST_TRAIN_BENCH_JOB_SCHEDULER:-none}"
export POST_TRAIN_BENCH_RESULTS_DIR="${POST_TRAIN_BENCH_RESULTS_DIR:-results}"
export POST_TRAIN_BENCH_EXPERIMENT_NAME="${POST_TRAIN_BENCH_EXPERIMENT_NAME:-}"

bash src/run_task_native.sh \
    "$EVALUATION_TASK" \
    "$AGENT" \
    "$MODEL_TO_TRAIN" \
    "$RUN_ID" \
    "$NUM_HOURS" \
    "$AGENT_CONFIG" \
    "$GPU_ID"

TASK_EXIT=$?

# --- Locate results ---
RESULT_PREFIX_SAFE=$(echo "$MODEL_TO_TRAIN" | tr '/:' '_')
AGENT_CONFIG_SAFE=$(echo "$AGENT_CONFIG" | tr '/:' '_')
EVAL_DIR="${REPO_ROOT}/results/${AGENT}_${AGENT_CONFIG_SAFE}_${NUM_HOURS}h/${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${RUN_ID}"

echo ""
echo "=== Committing experiment artifacts ==="

# --- Copy artifacts to experiment dir ---
for f in prompt.txt solve_parsed.txt time_taken.txt metrics.json; do
    if [ -f "${EVAL_DIR}/${f}" ]; then
        cp "${EVAL_DIR}/${f}" "${EXP_DIR}/${f}"
    fi
done

# Copy logs with .txt extension to avoid gitignore
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
import json, sys
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

# Read time taken
TIME_TAKEN="unknown"
if [ -f "${EXP_DIR}/time_taken.txt" ]; then
    TIME_TAKEN=$(cat "${EXP_DIR}/time_taken.txt")
fi

# Write status file
cat > "${EXP_DIR}/status.json" <<STAT
{
  "status": "${STATUS}",
  "exit_code": ${TASK_EXIT},
  "time_taken": "${TIME_TAKEN}",
  "completed_at": "$(date -u '+%Y-%m-%d %H:%M UTC')"
}
STAT

# --- Commit all artifacts ---
git add "${EXP_DIR}/"
git commit -m "Results: ${EVALUATION_TASK} / ${MODEL_SHORT} — ${STATUS} (${TIME_TAKEN})"
git push origin "$BRANCH_NAME" 2>&1

# --- Build artifacts list for PR ---
ARTIFACTS_LIST="- [\`config.json\`](experiments/${RUN_NAME}/config.json) — Run configuration"
ARTIFACTS_LIST+="\n- [\`status.json\`](experiments/${RUN_NAME}/status.json) — Final status"
[ -f "${EXP_DIR}/prompt.txt" ] && ARTIFACTS_LIST+="\n- [\`prompt.txt\`](experiments/${RUN_NAME}/prompt.txt) — System prompt given to agent"
[ -f "${EXP_DIR}/solve_parsed.txt" ] && ARTIFACTS_LIST+="\n- [\`solve_parsed.txt\`](experiments/${RUN_NAME}/solve_parsed.txt) — Full agent decision trajectory"
[ -f "${EXP_DIR}/metrics.json" ] && ARTIFACTS_LIST+="\n- [\`metrics.json\`](experiments/${RUN_NAME}/metrics.json) — Evaluation scores"
[ -f "${EXP_DIR}/time_taken.txt" ] && ARTIFACTS_LIST+="\n- [\`time_taken.txt\`](experiments/${RUN_NAME}/time_taken.txt) — Wall-clock duration"
[ -f "${EXP_DIR}/output_log.txt" ] && ARTIFACTS_LIST+="\n- [\`output_log.txt\`](experiments/${RUN_NAME}/output_log.txt) — Runner stdout"
[ -f "${EXP_DIR}/error_log.txt" ] && ARTIFACTS_LIST+="\n- [\`error_log.txt\`](experiments/${RUN_NAME}/error_log.txt) — Runner stderr"

# --- Update PR with final results ---
UPDATED_BODY="$(cat <<PRBODY
## Experiment: ${EVALUATION_TASK} / ${MODEL_SHORT} / ${CONFIG_SHORT}

### Config
| Parameter | Value |
|-----------|-------|
| Benchmark | \`${EVALUATION_TASK}\` |
| Model | \`${MODEL_TO_TRAIN}\` |
| Agent | \`${AGENT}\` (\`${AGENT_CONFIG}\`) |
| GPU | ${GPU_ID} ($(nvidia-smi --id=$GPU_ID --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)) |
| Time limit | ${NUM_HOURS}h |
| Started | $(date -u '+%Y-%m-%d %H:%M UTC') |
| Duration | ${TIME_TAKEN} |

### Status
${STATUS_EMOJI} **${STATUS_TEXT}**

### Results
| Metric | Value |
|--------|-------|
${METRICS_TABLE}

### Artifacts
$(echo -e "$ARTIFACTS_LIST")
PRBODY
)"

gh pr edit "$BRANCH_NAME" --body "$UPDATED_BODY" 2>&1

# Mark PR as ready if completed
if [ "$STATUS" = "completed" ]; then
    gh pr ready "$BRANCH_NAME" 2>&1
fi

echo ""
echo "=== Experiment complete ==="
echo "Status: ${STATUS_TEXT}"
echo "Time: ${TIME_TAKEN}"
echo "PR: $(cat ${EXP_DIR}/.pr_url 2>/dev/null || echo 'unknown')"
echo "Results: ${EXP_DIR}/"

# Return to main branch
git checkout main 2>/dev/null
