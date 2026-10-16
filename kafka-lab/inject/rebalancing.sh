#!/usr/bin/env bash
# A crash-looping member: consumer-1 dies after 20 s of uptime without leaving the group
# (like an OOM kill) and docker restarts it; its partitions wait for the session timeout,
# then the group rebalances, and again when it comes back.
source "$(dirname "$0")/../lib.sh"; knobs '{"crash_member": "consumer-1", "crash_every_s": 20}'
