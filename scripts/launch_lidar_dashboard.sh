#!/usr/bin/env bash
# Launch the MouseDroid telemetry stack locally for LiDAR dashboard validation.
#
# - Starts the mousedroid main loop in the foreground with mock hardware and
#   the rotating-wedge LiDAR pattern so the /lidar page shows obvious motion.
# - Prints the URLs to hit once the server reports healthy.
# - Assumes Grafana (3000) and Prometheus (9090) are already running via the
#   docker-compose stack; the script only launches the droid process.
#
# Usage:
#   ./scripts/launch_lidar_dashboard.sh            # mock hardware + rotating wedge
#   MOUSEDROID_PORT=8080 ./scripts/launch_lidar_dashboard.sh
#
# Press Ctrl-C to stop.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${MOUSEDROID_PORT:-8080}"
HOST="${MOUSEDROID_HOST:-127.0.0.1}"

# Prefer the local virtualenv if it exists.
if [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
    PY=".venv/Scripts/python.exe"
else
    PY="python3"
fi

echo "[launch_lidar_dashboard] using python: $PY"
echo "[launch_lidar_dashboard] telemetry: http://${HOST}:${PORT}/lidar"
echo "[launch_lidar_dashboard] metrics:   http://${HOST}:${PORT}/metrics"
echo "[launch_lidar_dashboard] grafana:   http://${HOST}:3000 (if running)"
echo "[launch_lidar_dashboard] prom:      http://${HOST}:9090 (if running)"
echo

"$PY" -m mousedroid.main \
    --config config/default.yaml config/local_lidar_validation.yaml
