#!/bin/bash
# Native (no-container) version of run_task.sh
# Usage: bash src/run_task_native.sh <eval> <agent> <model_to_train> <cluster_id> <num_hours> <agent_config> [gpu_id]
#
# Example:
#   bash src/run_task_native.sh gsm8k claude Qwen/Qwen3-1.7B-Base 001 10 claude-opus-4-6 0

set -eo pipefail

export EVALUATION_TASK="$1"
AGENT="$2"
MODEL_TO_TRAIN="$3"
CLUSTER_ID="$4"
NUM_HOURS="$5"
AGENT_CONFIG="$6"
GPU_ID="${7:-0}"

# Restrict to a single GPU
export CUDA_VISIBLE_DEVICES="$GPU_ID"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Pre-set defaults before sourcing set_env_vars.sh (which checks vars early)
export POST_TRAIN_BENCH_JOB_SCHEDULER="${POST_TRAIN_BENCH_JOB_SCHEDULER:-none}"
export POST_TRAIN_BENCH_RESULTS_DIR="${POST_TRAIN_BENCH_RESULTS_DIR:-results}"
export POST_TRAIN_BENCH_CONTAINERS_DIR="${POST_TRAIN_BENCH_CONTAINERS_DIR:-containers}"
export POST_TRAIN_BENCH_CONTAINER_NAME="${POST_TRAIN_BENCH_CONTAINER_NAME:-standard}"
export POST_TRAIN_BENCH_PROMPT="${POST_TRAIN_BENCH_PROMPT:-prompt}"
export POST_TRAIN_BENCH_EXPERIMENT_NAME="${POST_TRAIN_BENCH_EXPERIMENT_NAME:-}"

source src/commit_utils/set_env_vars.sh

# Activate the project venv
source "${REPO_ROOT}/.venv/bin/activate"

RESULT_PREFIX_SAFE=$(echo "$MODEL_TO_TRAIN" | tr '/:' '_')
AGENT_CONFIG_SAFE=$(echo "$AGENT_CONFIG" | tr '/:' '_')
RANDOM_UUID=$(uuidgen)

export EVAL_DIR="${POST_TRAIN_BENCH_RESULTS_DIR}/${AGENT}_${AGENT_CONFIG_SAFE}_${NUM_HOURS}h${POST_TRAIN_BENCH_EXPERIMENT_NAME}/${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${CLUSTER_ID}"

mkdir -p "${EVAL_DIR}"

# Log to files but also show on terminal
exec > >(tee "${EVAL_DIR}/output.log") 2> >(tee "${EVAL_DIR}/error.log" >&2)

echo "=== PostTrainBench Native Runner ==="
echo "Task: $EVALUATION_TASK | Agent: $AGENT | Model: $MODEL_TO_TRAIN | GPU: $GPU_ID"
echo "Config: $AGENT_CONFIG | Hours: $NUM_HOURS | Eval Dir: $EVAL_DIR"
echo "$@"

export TMP_SUBDIR="/tmp/posttrain_native_${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${RANDOM_UUID}"

JOB_DIR="${TMP_SUBDIR}/job_dir"
JOB_TMP="${TMP_SUBDIR}/tmp"

mkdir -p "${JOB_DIR}"
mkdir -p "${JOB_TMP}"

echo "Preparing job directory..."

mkdir -p "${JOB_DIR}/task"

# Copy evaluation files
cp "src/eval/tasks/${EVALUATION_TASK}/evaluate.py" "${JOB_DIR}/task"
if [ -d "src/eval/tasks/${EVALUATION_TASK}/evaluation_code" ]; then
    cp -r "src/eval/tasks/${EVALUATION_TASK}/evaluation_code" "${JOB_DIR}/task"
fi
cp -r src/eval/templates "${JOB_DIR}/task/"

if [ -d "src/eval/tasks/${EVALUATION_TASK}/task_context" ]; then
    cp -r "src/eval/tasks/${EVALUATION_TASK}/task_context/"* "${JOB_DIR}/task"
fi

# Copy codex config (even if not using codex, the dir is expected)
if [ -d "containers/other_home_data/.codex" ]; then
    cp -r "containers/other_home_data/.codex" "${JOB_DIR}/"
fi

BENCHMARK=$(cat "src/eval/tasks/${EVALUATION_TASK}/benchmark.txt")
PROMPT=$(python src/eval/general/get_prompt.py --model-to-train "$MODEL_TO_TRAIN" --benchmark-id "$EVALUATION_TASK" --num-hours "$NUM_HOURS" --agent "${AGENT}")
echo "$PROMPT" > "${EVAL_DIR}/prompt.txt"

bash src/utils/create_timer.sh "$NUM_HOURS" "${JOB_DIR}/task/timer.sh"

# Set API keys appropriately
export CODEX_API_KEY="${OPENAI_API_KEY:-}"
unset OPENAI_API_KEY 2>/dev/null || true
if [ "$EVALUATION_TASK" == "arenahardwriting" ] || [ "$EVALUATION_TASK" == "healthbench" ]; then
    export OPENAI_API_KEY="${CODEX_API_KEY}"
fi

# Copy agent scripts
cp src/utils/check_cuda.py "${JOB_DIR}/check_cuda.py"
cp src/utils/check_cuda_writing.py "${JOB_DIR}/check_cuda_writing.py"
cp "agents/${AGENT}/solve.sh" "${JOB_DIR}/agent_solve.sh"

# Copy agent-specific auth if present
if [ -f "agents/${AGENT}/auth.json" ]; then
    cp "agents/${AGENT}/auth.json" "${JOB_DIR}/.codex/auth.json"
fi
if [ -f "agents/${AGENT}/oauth_token" ]; then
    cp "agents/${AGENT}/oauth_token" "${JOB_DIR}/oauth_token"
fi

# Utils
with_record_the_time() {
    local begin=$(date --iso-8601=seconds)
    "$@"
    local exit_code=$?
    local end=$(date --iso-8601=seconds)

    local time_taken=$(( $(date --date="$end" +%s) - $(date --date="$begin" +%s) ))
    printf '%02d:%02d:%02d\n' \
        $(( time_taken / 3600 )) \
        $(( (time_taken % 3600) / 60 )) \
        $(( time_taken % 60 )) > "${EVAL_DIR}/time_taken.txt"

    return $exit_code
}

SOLVE_OUT="${EVAL_DIR}/solve_out.txt"

solve_task() {
    # Run agent natively (no container)
    # Set up environment to mimic the container
    export HF_HOME="${HF_HOME}"
    export VLLM_API_KEY="inspectai"
    export PYTHONNOUSERSITE="1"
    export PROMPT="${PROMPT}"
    export AGENT_CONFIG="${AGENT_CONFIG}"
    export TMPDIR="${JOB_TMP}"

    cd "${JOB_DIR}/task"

    timeout --signal=TERM --kill-after=30s "$((NUM_HOURS * 60 + 5))m" \
        bash -c "bash ${JOB_DIR}/agent_solve.sh" > "${SOLVE_OUT}" 2>&1
}

echo "================================"
echo "========= RUNNING TASK ========="
echo "================================"

with_record_the_time solve_task
SOLVE_EXIT=$?

# Return to repo root
cd "$REPO_ROOT"

echo "--- SOLVE DIAGNOSTICS ---"
echo "exit_code: $SOLVE_EXIT"
if [ $SOLVE_EXIT -eq 0 ]; then
    echo "status: exited normally"
elif [ $SOLVE_EXIT -eq 124 ]; then
    echo "status: killed by timeout (reached ${NUM_HOURS}h limit)"
elif [ $SOLVE_EXIT -gt 128 ]; then
    echo "status: killed by signal $((SOLVE_EXIT - 128)) ($(kill -l $((SOLVE_EXIT - 128)) 2>/dev/null || echo unknown))"
else
    echo "status: exited with error code $SOLVE_EXIT"
fi
echo "final_model_files: $(ls "${JOB_DIR}/task/final_model/" 2>/dev/null | wc -l)"
echo "hostname: $(hostname)"
echo "disk_job_dir: $(du -sh "${JOB_DIR}" 2>/dev/null | cut -f1)"
echo "disk_tmp: $(du -sh "${JOB_TMP}" 2>/dev/null | cut -f1)"
echo "memory: $(free -m 2>/dev/null | grep Mem | awk '{print "total=" $2 "MB used=" $3 "MB free=" $4 "MB"}')"
echo "--- END SOLVE DIAGNOSTICS ---"

echo "============================================"
echo "=== TASK COMPLETE, PARSING AGENT TRACE ==="
echo "============================================"

# Parse agent trace into human-readable format
TRACE_PARSER="agents/${AGENT}/human_readable_trace.py"
if [ -f "$TRACE_PARSER" ]; then
    python "$TRACE_PARSER" "${SOLVE_OUT}" -o "${EVAL_DIR}/solve_parsed.txt"
    cp "${EVAL_DIR}/solve_parsed.txt" "${JOB_DIR}/solve_parsed.txt"
else
    echo "Warning: No trace parser found at $TRACE_PARSER, using raw output"
    cp "${SOLVE_OUT}" "${JOB_DIR}/solve_parsed.txt"
fi

echo "============================="
echo "======== CLEANING UP ========"
echo "============================="

echo "Task directory contents:"
find "${JOB_DIR}/task" -maxdepth 2 -type f | head -50
echo "================================"

if [ -d "${JOB_DIR}/task/final_model" ]; then
    cp -r "${JOB_DIR}/task/final_model" "$EVAL_DIR/final_model"
fi

python containers/delete_hf_models.py "${JOB_DIR}/task" 2>/dev/null || true

cp -r "${JOB_DIR}/task" "$EVAL_DIR/task"

echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

export EVAL_COUNTER=0

run_evaluation() {
    local max_tokens_arg="$1"
    local eval_num="$2"
    # Kill any leftover GPU processes on our GPU
    nvidia-smi --id=$GPU_ID --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    sleep 5
    cd "${REPO_ROOT}/src/eval/tasks/${EVALUATION_TASK}"
    python evaluate.py \
        --model-path "$EVAL_DIR/final_model" \
        --templates-dir "${REPO_ROOT}/src/eval/templates" \
        --limit -1 \
        ${max_tokens_arg} \
        --json-output-file "${EVAL_DIR}/metrics.json" > "$EVAL_DIR/final_eval_${eval_num}.txt" 2>&1
    cd "$REPO_ROOT"
}

run_evaluation_with_retry() {
    local max_retries="$1"
    local max_tokens_arg="$2"

    for ((attempt=1; attempt<=max_retries; attempt++)); do
        sleep 5
        if [ -f "${EVAL_DIR}/metrics.json" ]; then
            return 0
        fi

        EVAL_COUNTER=$((EVAL_COUNTER + 1))
        export EVAL_COUNTER
        echo "Evaluation attempt $EVAL_COUNTER (phase attempt $attempt of $max_retries)"

        timeout --signal=TERM --kill-after=60s 28800s bash -c "
            source ${REPO_ROOT}/.venv/bin/activate
            export CUDA_VISIBLE_DEVICES=$GPU_ID
            export HF_HOME=${HF_HOME}
            export VLLM_API_KEY=inspectai
            export OPENAI_API_KEY=${OPENAI_API_KEY:-}
            cd ${REPO_ROOT}/src/eval/tasks/${EVALUATION_TASK}
            python evaluate.py \
                --model-path $EVAL_DIR/final_model \
                --templates-dir ${REPO_ROOT}/src/eval/templates \
                --limit -1 \
                $max_tokens_arg \
                --json-output-file ${EVAL_DIR}/metrics.json
        " > "$EVAL_DIR/final_eval_${EVAL_COUNTER}.txt" 2>&1 || true

        if [ -f "${EVAL_DIR}/metrics.json" ]; then
            return 0
        fi
    done

    return 1
}

if [ -d "${EVAL_DIR}/final_model" ]; then
    # First evaluation: up to 4 attempts
    run_evaluation_with_retry 4 ""

    # Second evaluation with adjusted max tokens
    case "${EVALUATION_TASK}" in
        aime2025)       MAX_TOKENS_ARG="--max-tokens 12000" ;;
        arenahardwriting) MAX_TOKENS_ARG="--max-new-tokens 12288" ;;
        bfcl)           MAX_TOKENS_ARG="--max-tokens 12000" ;;
        gpqamain)       MAX_TOKENS_ARG="--max-tokens 12000" ;;
        gsm8k)          MAX_TOKENS_ARG="--max-tokens 3000" ;;
        healthbench)    MAX_TOKENS_ARG="--max-new-tokens 12288" ;;
        humaneval)      MAX_TOKENS_ARG="--max-tokens 3000" ;;
        *)              MAX_TOKENS_ARG="" ;;
    esac
    run_evaluation_with_retry 3 "$MAX_TOKENS_ARG"

    # Third evaluation with further adjusted max tokens
    case "${EVALUATION_TASK}" in
        aime2025)       MAX_TOKENS_ARG="--max-tokens 8000" ;;
        arenahardwriting) MAX_TOKENS_ARG="--max-new-tokens 8192" ;;
        bfcl)           MAX_TOKENS_ARG="--max-tokens 8000" ;;
        gpqamain)       MAX_TOKENS_ARG="--max-tokens 8000" ;;
        gsm8k)          MAX_TOKENS_ARG="--max-tokens 2000" ;;
        healthbench)    MAX_TOKENS_ARG="--max-new-tokens 8192" ;;
        humaneval)      MAX_TOKENS_ARG="--max-tokens 2000" ;;
        *)              MAX_TOKENS_ARG="" ;;
    esac
    run_evaluation_with_retry 2 "$MAX_TOKENS_ARG"

    if [ -f "${EVAL_DIR}/metrics.json" ]; then
        echo "=== EVALUATION RESULTS ==="
        cat "${EVAL_DIR}/metrics.json"
    else
        echo "WARNING: Evaluation failed after all retry attempts"
    fi
else
    echo "WARNING: No final_model directory found - agent did not produce a model"
fi

echo "================================"
echo "======= ALL DONE ========"
echo "================================"
echo "Results saved to: ${EVAL_DIR}"
