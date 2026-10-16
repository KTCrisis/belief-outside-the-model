#!/usr/bin/env bash
# The consumers' GC logs, last 6 events each.
source "$(dirname "$0")/../lib.sh"
for c in consumer-1 consumer-2 consumer-3; do
  echo "$ tail gc-$c.log"; tail -6 "$LAB/logs/gc-$c.log" 2>/dev/null || echo "(no GC log)"
done
