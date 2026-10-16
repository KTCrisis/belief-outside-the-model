#!/usr/bin/env bash
# The order service sends ten times its usual rate.
source "$(dirname "$0")/../lib.sh"; knobs '{"rate": 3000}'
