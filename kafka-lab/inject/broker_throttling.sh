#!/usr/bin/env bash
# A default client quota caps every consumer's fetch rate at 15 kB/s.
source "$(dirname "$0")/../lib.sh"
kbin kafka-configs.sh --bootstrap-server $BS --alter --entity-type clients --entity-default --add-config consumer_byte_rate=15000 >/dev/null
