#!/usr/bin/env bash
# Client quotas on the broker, and the consumers' fetch throttle time.
source "$(dirname "$0")/../lib.sh"
echo "$ kafka-configs --describe --entity-type clients --entity-default"
q=$(kbin kafka-configs.sh --bootstrap-server $BS --describe --entity-type clients --entity-default 2>&1)
[ -n "$q" ] && echo "$q" || echo "(no default client quota)"
echo "$ consumer metrics, last report"
for c in consumer-1 consumer-2 consumer-3; do grep -h "fetch-throttle" "$LAB/logs/$c.log" 2>/dev/null | tail -1; done
