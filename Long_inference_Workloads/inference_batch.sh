#!/usr/bin/env bash
#
# llm_batch_runner.sh
#
# BATCH inference runner against a local Ollama-style API.
#
# Difference from a plain loop: each cycle fires the ENTIRE prompt set
# concurrently (not throttled a few at a time) so the Ollama server's
# own parallel scheduler can batch them together against the loaded
# model. This repeats as one cycle after another for a fixed duration.
#
# IMPORTANT — Ollama has no single-request "send N prompts, get N
# completions from one merged forward pass" API. True batching only
# happens when multiple requests arrive concurrently AND the server
# was started with enough parallel capacity. Before running this,
# set on the machine running `ollama serve`:
#
#   export OLLAMA_NUM_PARALLEL=10      # >= BATCH_CONCURRENCY below
#   export OLLAMA_MAX_LOADED_MODELS=1
#   export OLLAMA_MAX_QUEUE=512
#   ollama serve
#
# Without OLLAMA_NUM_PARALLEL raised, Ollama queues requests one at a
# time even though this script fires them all at once — you'll just
# see them processed sequentially instead of batched.
#
# Usage:
#   chmod +x llm_batch_runner.sh
#   ./llm_batch_runner.sh
#
# Run in the background for the full 3 hours with:
#   nohup ./llm_batch_runner.sh > /dev/null 2>&1 &
#
# Stop early with Ctrl+C (or `kill <pid>`) — finishes the in-flight
# batch cleanly, writes a final summary, then exits.

set -uo pipefail   # no -e: one failed request must not kill the loop

# ---------------------------- CONFIG ----------------------------
API_URL="http://127.0.0.1:8081/api/generate"
MODEL="qwen3:32b"
DURATION_SECONDS=$((3 * 60 * 60))   # 3 hours
BATCH_CONCURRENCY=2                  # keep low: 32b queues fast, high concurrency causes timeouts
REQUEST_TIMEOUT=900                  # seconds per request — 32b + thinking needs headroom
LOG_DIR="./llm_batch_logs"
# ------------------------------------------------------------------

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="${LOG_DIR}/${RUN_ID}"
SUMMARY_LOG="${RUN_LOG_DIR}/summary.log"
PROMPTS_FILE="${RUN_LOG_DIR}/prompts.txt"
RESPONSES_DIR="${RUN_LOG_DIR}/responses"
BATCH_JSONL="${RUN_LOG_DIR}/batch_results.jsonl"
JSONL_LOCK="${RUN_LOG_DIR}/.batch_results.lock"

mkdir -p "${RESPONSES_DIR}"
: > "${BATCH_JSONL}"
: > "${JSONL_LOCK}"

PROMPTS=(
"Explain what DNS is and why it's needed, in a few short paragraphs."
"What's the difference between TCP and UDP? Give a simple example of when you'd use each."
"Write a short Python function that checks if a number is prime, with a brief explanation."
"What is the difference between Agile and Waterfall? Summarize in a few sentences."
"Explain what a Docker container is and how it differs from a virtual machine."
"What is SQL injection and how can you prevent it? Keep it brief."
"Explain the difference between supervised and unsupervised machine learning with one example each."
"What is database indexing and why does it speed up queries? Explain simply."
"What is the difference between REST and GraphQL APIs? Summarize the key points."
"What are three common data structures used in coding interviews, and when would you use each?"
)

if [ "${BATCH_CONCURRENCY}" -eq 0 ]; then
  BATCH_CONCURRENCY=${#PROMPTS[@]}
fi

printf "%s\n" "${PROMPTS[@]}" > "${PROMPTS_FILE}"

# ---------------------- dependency checks --------------------------
if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required for safe JSON encoding/parsing. Install with: sudo apt-get install jq" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is required." >&2
  exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "ERROR: flock is required (util-linux). Install with: sudo apt-get install util-linux" >&2
  exit 1
fi

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SUMMARY_LOG}"
}

append_jsonl() {
  local line="$1"
  (
    flock -x 200
    echo "${line}" >> "${BATCH_JSONL}"
  ) 200>>"${JSONL_LOCK}"
}

# ----------------------- single request ----------------------------
run_one_prompt() {
  local prompt="$1" cycle="$2" idx="$3"
  local out_file="${RESPONSES_DIR}/cycle${cycle}_prompt${idx}.json"
  local req_log="${RESPONSES_DIR}/cycle${cycle}_prompt${idx}.log"
  local payload start end http_code duration ts

  payload=$(jq -n --arg model "${MODEL}" --arg prompt "${prompt}" \
    '{model: $model, prompt: $prompt, stream: false, think: false}')

  start=$(date +%s.%N)
  http_code=$(curl -s -o "${out_file}" -w "%{http_code}" \
    --max-time "${REQUEST_TIMEOUT}" \
    -H "Content-Type: application/json" \
    -d "${payload}" \
    "${API_URL}")
  end=$(date +%s.%N)
  duration=$(awk -v s="${start}" -v e="${end}" 'BEGIN{printf "%.2f", e-s}')
  ts="$(date '+%Y-%m-%d %H:%M:%S')"

  local stats="" ttft="" tps=""
  if [ "${http_code}" = "200" ] && jq -e . "${out_file}" >/dev/null 2>&1; then
    stats=$(jq -r '
      "time_to_first_token_s: " + (((.load_duration + .prompt_eval_duration)/1e9)|tostring),
      "prompt_eval_count: " + (.prompt_eval_count|tostring),
      "prompt_eval_duration_s: " + ((.prompt_eval_duration/1e9)|tostring),
      "eval_count: " + (.eval_count|tostring),
      "eval_duration_s: " + ((.eval_duration/1e9)|tostring),
      "tokens_per_sec_gen: " + ((.eval_count / (.eval_duration/1e9))|tostring),
      "total_duration_s: " + ((.total_duration/1e9)|tostring)
    ' "${out_file}" 2>/dev/null)
    ttft=$(echo "${stats}" | awk -F': ' '/time_to_first_token_s/{printf "%.3f", $2}')
    tps=$(echo "${stats}" | awk -F': ' '/tokens_per_sec_gen/{printf "%.2f", $2}')
  fi

  {
    echo "timestamp: ${ts}"
    echo "cycle: ${cycle}  prompt_index: ${idx}"
    echo "http_code: ${http_code}  duration_seconds: ${duration}"
    echo "prompt_preview: ${prompt:0:80}..."
    echo "response_file: ${out_file}"
    [ -n "${stats}" ] && echo "${stats}"
  } > "${req_log}"

  # One JSONL line per request, appended into the consolidated batch file
  local jsonl_line
  jsonl_line=$(jq -nc \
    --arg ts "${ts}" --argjson cycle "${cycle}" --argjson idx "${idx}" \
    --arg model "${MODEL}" --arg http_code "${http_code}" \
    --arg duration "${duration}" --arg ttft "${ttft:-null}" --arg tps "${tps:-null}" \
    --arg prompt_preview "${prompt:0:120}" --arg response_file "${out_file}" \
    '{timestamp:$ts, cycle:$cycle, prompt_index:$idx, model:$model,
      http_code:($http_code|tonumber? // $http_code),
      duration_s:($duration|tonumber? // null),
      ttft_s:($ttft|tonumber? // null),
      tokens_per_sec:($tps|tonumber? // null),
      prompt_preview:$prompt_preview, response_file:$response_file}')
  append_jsonl "${jsonl_line}"

  local status_line
  if [ "${http_code}" = "200" ]; then
    status_line="OK    cycle=${cycle} prompt=${idx} total=${duration}s ttft=${ttft}s tok/s=${tps} -> ${out_file}"
  else
    status_line="FAIL  cycle=${cycle} prompt=${idx} http=${http_code} ${duration}s -> see ${req_log}"
  fi
  echo "[${ts}] ${status_line}" >> "${SUMMARY_LOG}"
  echo "${status_line}"
}

# ----------------------- one batch (= one cycle) ----------------------
run_batch() {
  local cycle="$1"
  local pids=()
  for i in "${!PROMPTS[@]}"; do
    run_one_prompt "${PROMPTS[$i]}" "${cycle}" "${i}" &
    pids+=($!)
    if (( ${#pids[@]} >= BATCH_CONCURRENCY )); then
      wait -n
    fi
  done
  wait
}

# ------------------------------ main ----------------------------------
START_TS=$(date +%s)
END_TS=$((START_TS + DURATION_SECONDS))
CYCLE=0
INTERRUPTED=0

trap 'log "Interrupted by user — finishing current batch, then stopping."; INTERRUPTED=1' INT TERM

log "=== Starting batch inference run ==="
log "API: ${API_URL}  Model: ${MODEL}  Duration: ${DURATION_SECONDS}s  Batch size: ${#PROMPTS[@]}  Concurrency: ${BATCH_CONCURRENCY}"
log "Logs directory: ${RUN_LOG_DIR}"
log "Consolidated results: ${BATCH_JSONL}"
log "Reminder: confirm OLLAMA_NUM_PARALLEL on the server is >= ${BATCH_CONCURRENCY} for true concurrent batching."

while [ "$(date +%s)" -lt "${END_TS}" ] && [ "${INTERRUPTED}" -eq 0 ]; do
  CYCLE=$((CYCLE + 1))
  CYCLE_START=$(date +%s)
  log "--- Batch ${CYCLE} starting (${#PROMPTS[@]} prompts, concurrency=${BATCH_CONCURRENCY}) ---"

  run_batch "${CYCLE}"

  CYCLE_END=$(date +%s)
  REMAINING=$((END_TS - CYCLE_END))
  (( REMAINING < 0 )) && REMAINING=0
  log "--- Batch ${CYCLE} finished in $((CYCLE_END - CYCLE_START))s | time remaining: ${REMAINING}s ---"
done

# ----------------------- final aggregate stats -------------------------
if [ -s "${BATCH_JSONL}" ]; then
  AGG=$(jq -s '
    def safe_avg(f): [.[] | f] | select(length > 0) | add / length;
    {
      total_requests: length,
      successful: ([.[] | select(.http_code == 200)] | length),
      failed: ([.[] | select(.http_code != 200)] | length),
      avg_ttft_s: (try safe_avg(select(.ttft_s != null and .ttft_s > 0) | .ttft_s) catch null),
      avg_tokens_per_sec: (try safe_avg(select(.tokens_per_sec != null and .tokens_per_sec > 0) | .tokens_per_sec) catch null)
    }' "${BATCH_JSONL}")
  log "=== Aggregate stats ==="
  log "$(echo "${AGG}" | jq -c .)"
fi

log "=== Run complete. Total batches: ${CYCLE} ==="
log "Per-prompt files: ${RESPONSES_DIR}"
log "Consolidated JSONL: ${BATCH_JSONL}"
log "Summary log: ${SUMMARY_LOG}"
