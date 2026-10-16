#!/usr/bin/env bash
# Between episodes: clear the fault, start from a fresh topic and group (no offsets
# or lag carried over from an earlier episode), restart the apps on empty logs.
source "$(dirname "$0")/lib.sh"
"$LAB/inject/clear.sh"
$DC stop -t 5 consumer-1 consumer-2 consumer-3 producer >/dev/null 2>&1
# the group can only be deleted once the broker sees it empty (members may linger until their session ends)
for _ in $(seq 60); do
  state=$(kbin kafka-consumer-groups.sh --bootstrap-server $BS --describe --group lab --state 2>/dev/null | awk 'NR>2 {print $(NF-1)}' | head -1)
  { [ "$state" = "Empty" ] || [ "$state" = "Dead" ] || [ -z "$state" ]; } && break
  sleep 1
done
kbin kafka-consumer-groups.sh --bootstrap-server $BS --delete --group lab >/dev/null 2>&1
kbin kafka-topics.sh --bootstrap-server $BS --delete --topic orders >/dev/null 2>&1
for _ in $(seq 30); do
  kbin kafka-topics.sh --bootstrap-server $BS --list 2>/dev/null | grep -qx orders || break
  sleep 1
done
if ! kbin kafka-topics.sh --bootstrap-server $BS --create --topic orders --partitions 6 --replication-factor 1 >/dev/null 2>&1; then
  echo "reset.sh: could not recreate topic orders (group state was: ${state:-unknown})" >&2
  exit 1
fi
rm -f "$LAB"/logs/*.log*
$DC start producer consumer-1 consumer-2 consumer-3 >/dev/null 2>&1
