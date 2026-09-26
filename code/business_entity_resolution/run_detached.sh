#!/usr/bin/env bash
# Launch a pipeline command in its own session so it survives the terminal/agent exiting.
# Usage: bash code/business_entity_resolution/run_detached.sh <log> <pipeline args...>
set -euo pipefail
LOG=$1; shift
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../../.venv/bin/python"
nohup "$PY" -c 'import os, sys; os.setsid(); os.execv(sys.argv[1], sys.argv[1:])' \
  "$PY" -W ignore "$HERE/src/pipeline.py" "$@" > "$LOG" 2>&1 < /dev/null &
echo "started pid $! -> $LOG"
