#!/usr/bin/env bash
# Second lane after the restart: the hold reruns (amendment A1) first, then any remaining main cells.
# Both lanes share one budget over the main and hold state roots and poll the lease every 2 s.
W=$(cd "$(dirname "$0")/../../../.." && pwd)
R=${1:-/ml2/hypergan/dig2-20261016/wallclock-reruns}
export LEASE=$W/reports/forge/dig2-20261016/wallclock-reruns/fastlease.sh
L=$W/reports/forge/dig2-20261016/wallclock-reruns/lane.sh
bash "$L" "$R/hold" "$W" 7200 "$R $R/hold"
bash "$L" "$R" "$W" 7200 "$R $R/hold"
