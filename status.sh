#!/bin/bash
# Quick status check for context-bench loop v5
LOG=/root/code/context-bench/loop_results_v5/run.log
WD=/root/code/context-bench/loop_results_v5/watchdog.log
PID=$(cat /root/code/context-bench/loop_results_v5/loop.pid 2>/dev/null)

echo "=== Loop v5 Status $(date '+%H:%M:%S') ==="
echo "Loop PID: $PID  alive=$(kill -0 $PID 2>/dev/null && echo YES || echo NO)"
echo ""
echo "--- Last 20 lines of run.log ---"
tail -20 "$LOG" 2>/dev/null
echo ""
echo "--- Last 5 lines of watchdog.log ---"
tail -5 "$WD" 2>/dev/null
echo ""
echo "--- Active relay connections ---"
ss -tp | grep 18082 | head -5
