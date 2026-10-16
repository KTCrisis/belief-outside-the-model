#!/usr/bin/env bash
# Incoming rate per partition, from the log-end offsets 10 s apart.
source "$(dirname "$0")/../lib.sh"
a=$(kbin kafka-get-offsets.sh --bootstrap-server $BS --topic orders 2>/dev/null); sleep 10
b=$(kbin kafka-get-offsets.sh --bootstrap-server $BS --topic orders 2>/dev/null)
echo "$ incoming messages per second on orders, per partition (10 s window)"
join -t: -1 2 -2 2 <(echo "$a" | sort -t: -k2,2) <(echo "$b" | sort -t: -k2,2) |
  awk -F: '{r=($5-$3)/10; s+=r; printf "partition %s: %.0f msg/s\n", $1, r} END {printf "total: %.0f msg/s\n", s}'
