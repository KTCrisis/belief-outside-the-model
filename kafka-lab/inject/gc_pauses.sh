#!/usr/bin/env bash
# Memory pressure on 512 MB heaps: a 470 MB live cache plus heavy churn, so the serial collector runs long full GCs back to back.
source "$(dirname "$0")/../lib.sh"; knobs '{"gc_live_mb": 470, "alloc_mb_per_s": 300}'
