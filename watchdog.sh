#!/bin/bash
# watchdog.sh — monitors the autoresearch loop and relay, restarts on crash
# Usage: nohup bash watchdog.sh > loop_results/watchdog.log 2>&1 &

set -euo pipefail

LOOP_LOG="/root/code/context-bench/loop_results/run.log"
WATCHDOG_LOG="/root/code/context-bench/loop_results/watchdog.log"
RESULTS_DIR="/root/code/context-bench/loop_results"
RELAY_DIR="/root/code/claude-relay"
RELAY_URL="http://localhost:18082"
LOOP_CMD=(
    python3 /root/code/context-bench/loop.py
    --relay "$RELAY_URL"
    --model claude-haiku-4-5-20251001
    --dataset locomo
    --iterations 40
    --eval-n 4
    --max-qa-per-conv 8
    --seed 42
    --output-dir "$RESULTS_DIR"
    --resume
)

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$WATCHDOG_LOG"
}

check_relay() {
    curl -s --max-time 10 "$RELAY_URL/v1/models" > /dev/null 2>&1
}

start_relay() {
    log "Starting relay..."
    cd "$RELAY_DIR"
    nohup uv run agent-relay serve --port 18082 --max-concurrent 4 \
        >> "$RESULTS_DIR/relay.log" 2>&1 &
    sleep 10
    if check_relay; then
        log "Relay started OK"
    else
        log "ERROR: relay failed to start"
        return 1
    fi
}

start_loop() {
    log "Starting loop (resume=$RESUME_FLAG)..."
    cd /root/code/context-bench
    nohup "${LOOP_CMD[@]}" >> "$LOOP_LOG" 2>&1 &
    LOOP_PID=$!
    log "Loop PID=$LOOP_PID"
    echo "$LOOP_PID" > "$RESULTS_DIR/loop.pid"
}

# ---- Initial state ----
RESUME_FLAG=""
mkdir -p "$RESULTS_DIR"
log "=== Watchdog started ==="

# Relay: ensure it's up
if ! check_relay; then
    log "Relay not responding, starting..."
    start_relay || { log "FATAL: cannot start relay"; exit 1; }
else
    log "Relay already up"
fi

# Check if loop is already running (passed in from parent)
EXISTING_PID=""
if [[ -f "$RESULTS_DIR/loop.pid" ]]; then
    EXISTING_PID=$(cat "$RESULTS_DIR/loop.pid" 2>/dev/null || echo "")
fi

if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    LOOP_PID="$EXISTING_PID"
    log "Loop already running at PID=$LOOP_PID"
else
    # Check if there's a checkpoint to resume from
    if [[ -f "$RESULTS_DIR/best_pipeline.py" ]]; then
        RESUME_FLAG="--resume"
        log "Checkpoint found, will resume"
    fi
    start_loop
fi

# ---- Monitor loop ----
STALL_COUNT=0
LAST_LOG_SIZE=0
MAX_STALL=6   # 6 × 10min = 60min stall before restart

while true; do
    sleep 600  # check every 10 minutes

    # 1. Check relay health
    if ! check_relay; then
        log "WARNING: relay down, restarting..."
        start_relay || { log "ERROR: relay restart failed, will retry next cycle"; continue; }
    fi

    # 2. Check loop process
    if ! kill -0 "$LOOP_PID" 2>/dev/null; then
        log "Loop process $LOOP_PID is gone"

        # Check if it completed normally (log ends with "Loop complete")
        if grep -q "Loop complete" "$LOOP_LOG" 2>/dev/null; then
            log "Loop completed normally. Watchdog exiting."
            exit 0
        fi

        # Crashed — restart from checkpoint
        log "Loop crashed, restarting from checkpoint..."
        RESUME_FLAG="--resume"
        start_loop
        STALL_COUNT=0
        LAST_LOG_SIZE=0
        continue
    fi

    # 3. Stall detection — log file not growing
    CURRENT_SIZE=$(wc -c < "$LOOP_LOG" 2>/dev/null || echo 0)
    if [[ "$CURRENT_SIZE" -eq "$LAST_LOG_SIZE" ]]; then
        STALL_COUNT=$((STALL_COUNT + 1))
        log "WARNING: log stalled for $((STALL_COUNT * 10)) min (PID=$LOOP_PID alive)"
        if [[ "$STALL_COUNT" -ge "$MAX_STALL" ]]; then
            log "Stall threshold reached (${MAX_STALL}×10min), killing and restarting..."
            kill -9 "$LOOP_PID" 2>/dev/null || true
            sleep 2
            RESUME_FLAG="--resume"
            start_loop
            STALL_COUNT=0
            LAST_LOG_SIZE=0
        fi
    else
        STALL_COUNT=0
        LAST_LOG_SIZE="$CURRENT_SIZE"
        # Print last iteration line for visibility
        LAST_ITER=$(grep "Iter " "$LOOP_LOG" 2>/dev/null | tail -1 || echo "(no iter yet)")
        log "OK — $LAST_ITER"
    fi
done
