#!/usr/bin/env bash
# Back to healthy: default knobs, no client quota.
source "$(dirname "$0")/../lib.sh"; knobs '{}'
kbin kafka-configs.sh --bootstrap-server $BS --alter --entity-type clients --entity-default --delete-config consumer_byte_rate >/dev/null 2>&1 || true
