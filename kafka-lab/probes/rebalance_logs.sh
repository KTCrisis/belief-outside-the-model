#!/usr/bin/env bash
# Group coordination events in the consumers' logs, last 2 minutes.
source "$(dirname "$0")/../lib.sh"
out=$($DC logs --no-log-prefix --since 2m consumer-1 consumer-2 consumer-3 2>/dev/null | grep -E "Coordinator" | tail -40)
echo "$ grep Coordinator consumer logs (last 2 min)"
[ -n "$out" ] && echo "$out" || echo "(no group coordination events in the last 2 minutes)"
