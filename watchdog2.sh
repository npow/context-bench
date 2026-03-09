#!/bin/bash
LOOP_LOG="/root/code/context-bench/loop_results_v2/run.log"
WATCHDOG_LOG="/root/code/context-bench/loop_results_v2/watchdog.log"
RESULTS_DIR="/root/code/context-bench/loop_results_v2"
RELAY_DIR="/root/code/claude-relay"
RELAY_URL="http://localhost:18082"
LOOP_CMD=(
    python3 /root/code/context-bench/loop.py
    --relay "$RELAY_URL"
    --model claude-haiku-4-5-20251001
    --dataset locomo
    --iterations 50
    --eval-n 4
    --max-qa-per-conv 10
    --seed 42
    --output-dir "$RESULTS_DIR"
    --resume
)

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$WATCHDOG_LOG"; }

check_relay() { curl -s --max-time 10 "$RELAY_URL/v1/models" > /dev/null 2>&1; }

start_relay() {
    log "Starting relay..."
    cd "$RELAY_DIR"
    nohup uv run agent-relay serve --port 18082 --max-concurrent 4 \
        >> "$RESULTS_DIR/relay.log" 2>&1 &
    sleep 10
    check_relay && log "Relay started OK" || { log "ERROR: relay failed"; return 1; }
}

start_loop() {
    log "Starting loop (resuming from checkpoint)..."
    cd /root/code/context-bench
    nohup "${LOOP_CMD[@]}" >> "$LOOP_LOG" 2>&1 &
    LOOP_PID=$!
    echo "$LOOP_PID" > "$RESULTS_DIR/loop.pid"
    log "Loop PID=$LOOP_PID"
}

mkdir -p "$RESULTS_DIR"
log "=== Watchdog v2 started ==="

check_relay || start_relay || { log "FATAL: cannot start relay"; exit 1; }

LOOP_PID=$(cat "$RESULTS_DIR/loop.pid" 2>/dev/null || echo "")
if [[ -n "$LOOP_PID" ]] && kill -0 "$LOOP_PID" 2>/dev/null; then
    log "Loop already running at PID=$LOOP_PID"
else
    start_loop
fi

STALL_COUNT=0; LAST_LOG_SIZE=0; MAX_STALL=6
while true; do
    sleep 600
    check_relay || start_relay || continue

    if ! kill -0 "$LOOP_PID" 2>/dev/null; then
        grep -q "Loop complete" "$LOOP_LOG" 2>/dev/null && { log "Loop completed. Exiting."; exit 0; }
        log "Loop crashed, restarting..."
        start_loop; STALL_COUNT=0; LAST_LOG_SIZE=0; continue
    fi

    # Also check for completion even when process is still alive (it may linger)
    grep -q "Loop complete" "$LOOP_LOG" 2>/dev/null && { log "Loop completed normally. Exiting."; exit 0; }

    CURRENT_SIZE=$(wc -c < "$LOOP_LOG" 2>/dev/null || echo 0)
    if [[ "$CURRENT_SIZE" -eq "$LAST_LOG_SIZE" ]]; then
        STALL_COUNT=$((STALL_COUNT+1))
        log "Stalled ${STALL_COUNT}×10min"
        if [[ "$STALL_COUNT" -ge "$MAX_STALL" ]]; then
            log "Stall limit — killing and restarting"
            kill -9 "$LOOP_PID" 2>/dev/null; sleep 2
            start_loop; STALL_COUNT=0; LAST_LOG_SIZE=0
        fi
    else
        STALL_COUNT=0; LAST_LOG_SIZE="$CURRENT_SIZE"
        log "OK — $(grep 'Iter ' "$LOOP_LOG" 2>/dev/null | tail -1)"
    fi
done
