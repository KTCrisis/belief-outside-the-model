#!/usr/bin/env bash
# 98 % of the orders are pinned to partition 0 (a hot key).
source "$(dirname "$0")/../lib.sh"; knobs '{"skew": 0.98}'
