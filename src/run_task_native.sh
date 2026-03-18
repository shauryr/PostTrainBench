#!/bin/bash
# Native (no-container) version of run_task.sh
# Uses EVAL_DIR as working directory (no /tmp, no containers)
#
# Usage: bash src/run_task_native.sh <eval> <agent> <model_to_train> <cluster_id> <num_hours> <agent_config> [gpu_id]
#
# Example:
#   bash src/run_task_native.sh gsm8k claude Qwen/Qwen3-1.7B-Base 001 10 claude-opus-4-6 0

set -o pipefail

# --- Argument validation ---
if [ $# -lt 6 ]; then
    echo "Usage: $0 <eval> <agent> <model> <cluster_id> <hours> <config> [gpu_id]" >&2
    exit 1
fi

export EVALUATION_TASK="$1"
AGENT="$2"
MODEL_TO_TRAIN="$3"
CLUSTER_ID="$4"
NUM_HOURS="$5"
AGENT_CONFIG="$6"
GPU_ID="${7:-all}"

if ! [[ "$NUM_HOURS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: NUM_HOURS must be a positive integer, got '$NUM_HOURS'" >&2
    exit 1
fi

# GPU visibility: "all" uses all GPUs, otherwise restrict to specified GPUs
if [ "$GPU_ID" = "all" ]; then
    # Don't set CUDA_VISIBLE_DEVICES — let agent see all GPUs
    unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
else
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Pre-set defaults
export POST_TRAIN_BENCH_JOB_SCHEDULER="${POST_TRAIN_BENCH_JOB_SCHEDULER:-none}"
export POST_TRAIN_BENCH_RESULTS_DIR="${POST_TRAIN_BENCH_RESULTS_DIR:-results}"
export POST_TRAIN_BENCH_CONTAINERS_DIR="${POST_TRAIN_BENCH_CONTAINERS_DIR:-containers}"
export POST_TRAIN_BENCH_CONTAINER_NAME="${POST_TRAIN_BENCH_CONTAINER_NAME:-standard}"
export POST_TRAIN_BENCH_PROMPT="${POST_TRAIN_BENCH_PROMPT:-prompt_native}"
export POST_TRAIN_BENCH_EXPERIMENT_NAME="${POST_TRAIN_BENCH_EXPERIMENT_NAME:-}"

source src/commit_utils/set_env_vars.sh

# Activate the project venv
source "${REPO_ROOT}/.venv/bin/activate"

# Python.h headers needed by vLLM for cuda_utils compilation
export C_INCLUDE_PATH="${HOME}/.local/include/python3.10:${HOME}/.local/include:${C_INCLUDE_PATH:-}"
export CPATH="${HOME}/.local/include/python3.10:${HOME}/.local/include:${CPATH:-}"

RESULT_PREFIX_SAFE=$(echo "$MODEL_TO_TRAIN" | tr '/:' '_')
AGENT_CONFIG_SAFE=$(echo "$AGENT_CONFIG" | tr '/:' '_')

# Use absolute path — this IS the working directory (no /tmp)
export EVAL_DIR="${REPO_ROOT}/${POST_TRAIN_BENCH_RESULTS_DIR}/${AGENT}_${AGENT_CONFIG_SAFE}_${NUM_HOURS}h${POST_TRAIN_BENCH_EXPERIMENT_NAME}/${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${CLUSTER_ID}"
TASK_DIR="${EVAL_DIR}/task"

mkdir -p "${TASK_DIR}"

# Log to files but also show on terminal
exec > >(tee "${EVAL_DIR}/output.log") 2> >(tee "${EVAL_DIR}/error.log" >&2)

echo "=== PostTrainBench Native Runner ==="
echo "Task: $EVALUATION_TASK | Agent: $AGENT | Model: $MODEL_TO_TRAIN | GPU: $GPU_ID"
echo "Config: $AGENT_CONFIG | Hours: $NUM_HOURS"
echo "Working dir: $TASK_DIR"

# --- Prepare task directory ---
echo "Preparing task directory..."

cp "src/eval/tasks/${EVALUATION_TASK}/evaluate.py" "${TASK_DIR}/"
if [ -d "src/eval/tasks/${EVALUATION_TASK}/evaluation_code" ]; then
    cp -r "src/eval/tasks/${EVALUATION_TASK}/evaluation_code" "${TASK_DIR}/"
fi
cp -r src/eval/templates "${TASK_DIR}/"

if [ -d "src/eval/tasks/${EVALUATION_TASK}/task_context" ]; then
    cp -r "src/eval/tasks/${EVALUATION_TASK}/task_context/." "${TASK_DIR}/" 2>/dev/null || true
fi

# --- Copy seed directory if it exists for this benchmark ---
if [ -d "seed/${EVALUATION_TASK}" ]; then
    echo "Seeding with previous findings from seed/${EVALUATION_TASK}/"
    cp -r "seed/${EVALUATION_TASK}" "${TASK_DIR}/previous_run"
    # Use seeded prompt if available
    if [ -z "${POST_TRAIN_BENCH_PROMPT:-}" ] || [ "${POST_TRAIN_BENCH_PROMPT}" = "prompt_native" ]; then
        if [ -f "src/eval/general/prompt_native_seeded.txt" ]; then
            export POST_TRAIN_BENCH_PROMPT="prompt_native_seeded"
        fi
    fi
fi

# --- Generate prompt (use native prompt template) ---
BENCHMARK=$(cat "src/eval/tasks/${EVALUATION_TASK}/benchmark.txt")
PROMPT=$(python src/eval/general/get_prompt.py --model-to-train "$MODEL_TO_TRAIN" --benchmark-id "$EVALUATION_TASK" --num-hours "$NUM_HOURS" --agent "${AGENT}")
echo "$PROMPT" > "${EVAL_DIR}/prompt.txt"

bash src/utils/create_timer.sh "$NUM_HOURS" "${TASK_DIR}/timer.sh"

# --- Auth handling ---
# Auto-detect OAuth token vs API key for Claude
if [[ "${ANTHROPIC_API_KEY:-}" == sk-ant-oat* ]]; then
    echo "Detected OAuth token — setting CLAUDE_CODE_OAUTH_TOKEN"
    export CLAUDE_CODE_OAUTH_TOKEN="${ANTHROPIC_API_KEY}"
    export ANTHROPIC_API_KEY=""
fi

export CODEX_API_KEY="${OPENAI_API_KEY:-}"
unset OPENAI_API_KEY 2>/dev/null || true
if [ "$EVALUATION_TASK" == "arenahardwriting" ] || [ "$EVALUATION_TASK" == "healthbench" ]; then
    export OPENAI_API_KEY="${CODEX_API_KEY}"
fi

# --- Copy agent solve script ---
cp "agents/${AGENT}/solve.sh" "${EVAL_DIR}/agent_solve.sh"

if [ -f "agents/${AGENT}/auth.json" ]; then
    mkdir -p "${EVAL_DIR}/.codex"
    cp "agents/${AGENT}/auth.json" "${EVAL_DIR}/.codex/auth.json"
fi
if [ -f "agents/${AGENT}/oauth_token" ]; then
    cp "agents/${AGENT}/oauth_token" "${EVAL_DIR}/oauth_token"
fi

# --- Time tracking ---
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
    export VLLM_API_KEY="inspectai"
    export PYTHONNOUSERSITE="1"
    export PROMPT="${PROMPT}"
    export AGENT_CONFIG="${AGENT_CONFIG}"
    # Short TMPDIR for Unix sockets (108-char limit)
    export TMPDIR="/tmp/ptb_$(echo $CLUSTER_ID | cut -c1-8)"
    mkdir -p "$TMPDIR"
    # Ensure .venv python is used by all subprocesses (including Claude's tool calls)
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
    export VIRTUAL_ENV="${REPO_ROOT}/.venv"
    # Claude Code temp/working files go into the experiment dir for tracking
    export CLAUDE_CODE_TMPDIR="${EVAL_DIR}"

    cd "${TASK_DIR}"

    timeout --signal=TERM --kill-after=30s "$((NUM_HOURS * 60 + 5))m" \
        bash -c "export PATH='${REPO_ROOT}/.venv/bin:${PATH}'; export VIRTUAL_ENV='${REPO_ROOT}/.venv'; export CLAUDE_CODE_TMPDIR='${EVAL_DIR}'; export C_INCLUDE_PATH='${HOME}/.local/include/python3.10:${HOME}/.local/include'; export CPATH='${HOME}/.local/include/python3.10:${HOME}/.local/include'; bash ${EVAL_DIR}/agent_solve.sh" > "${SOLVE_OUT}" 2>&1
}

echo "================================"
echo "========= RUNNING TASK ========="
echo "================================"

with_record_the_time solve_task && SOLVE_EXIT=0 || SOLVE_EXIT=$?

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
echo "final_model_files: $(ls "${TASK_DIR}/final_model/" 2>/dev/null | wc -l)"
echo "hostname: $(hostname)"
echo "disk_task_dir: $(du -sh "${TASK_DIR}" 2>/dev/null | cut -f1)"
echo "memory: $(free -m 2>/dev/null | grep Mem | awk '{print "total=" $2 "MB used=" $3 "MB free=" $4 "MB"}')"
echo "--- END SOLVE DIAGNOSTICS ---"

echo "============================================"
echo "=== TASK COMPLETE, PARSING AGENT TRACE ==="
echo "============================================"

TRACE_PARSER="${REPO_ROOT}/agents/${AGENT}/human_readable_trace.py"
if [ -f "$TRACE_PARSER" ]; then
    python "$TRACE_PARSER" "${SOLVE_OUT}" -o "${EVAL_DIR}/solve_parsed.txt" || true
else
    echo "Warning: No trace parser found at $TRACE_PARSER, using raw output"
    cp "${SOLVE_OUT}" "${EVAL_DIR}/solve_parsed.txt" 2>/dev/null || true
fi

# Move final_model to EVAL_DIR level if agent put it in task/
if [ -d "${TASK_DIR}/final_model" ] && [ ! -d "${EVAL_DIR}/final_model" ]; then
    mv "${TASK_DIR}/final_model" "${EVAL_DIR}/final_model"
fi

echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

export EVAL_COUNTER=0
export VLLM_API_KEY="inspectai"

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

        (
            source "${REPO_ROOT}/.venv/bin/activate"
            # Use single GPU for eval (vLLM serves on 1 GPU)
            export CUDA_VISIBLE_DEVICES="${GPU_ID%%,*}"
            [ "$GPU_ID" = "all" ] && export CUDA_VISIBLE_DEVICES="0"
            cd "${REPO_ROOT}/src/eval/tasks/${EVALUATION_TASK}"
            timeout --signal=TERM --kill-after=60s 28800s \
                python evaluate.py \
                    --model-path "$EVAL_DIR/final_model" \
                    --templates-dir "${REPO_ROOT}/src/eval/templates" \
                    --limit -1 \
                    $max_tokens_arg \
                    --json-output-file "${EVAL_DIR}/metrics.json"
        ) > "$EVAL_DIR/final_eval_${EVAL_COUNTER}.txt" 2>&1 || true

        if [ -f "${EVAL_DIR}/metrics.json" ]; then
            return 0
        fi
    done
    return 1
}

if [ -d "${EVAL_DIR}/final_model" ]; then
    run_evaluation_with_retry 4 ""

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

echo "EVAL_DIR=${EVAL_DIR}" > "${EVAL_DIR}/.eval_dir"

echo "================================"
echo "======= ALL DONE ========"
echo "================================"
echo "Results saved to: ${EVAL_DIR}"
