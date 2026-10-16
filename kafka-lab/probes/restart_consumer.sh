#!/usr/bin/env bash
# Restart one consumer and watch the total lag for 30 s.
source "$(dirname "$0")/../lib.sh"
echo "$ total lag before restart: $(total_lag)"
$DC restart -t 5 consumer-1 >/dev/null 2>&1
for t in 10 20 30; do sleep 10; echo "$ total lag ${t} s after restart: $(total_lag)"; done
