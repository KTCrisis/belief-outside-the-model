#!/usr/bin/env bash
# Per-record processing becomes five times slower (CPU-bound enrichment).
source "$(dirname "$0")/../lib.sh"; knobs '{"process_ms": 20}'
