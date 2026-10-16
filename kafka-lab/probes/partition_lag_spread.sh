#!/usr/bin/env bash
# Lag per partition for the consumer group.
source "$(dirname "$0")/../lib.sh"
echo "$ kafka-consumer-groups --describe --group lab"
kbin kafka-consumer-groups.sh --bootstrap-server $BS --describe --group lab 2>&1 | sed '/^$/d'
