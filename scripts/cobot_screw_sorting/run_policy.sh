#!/usr/bin/env bash
set -euo pipefail
source /opt/ros/noetic/setup.bash
source client/.venv/bin/activate
export PYTHONPATH=/home/xlk/work/openpi:${PYTHONPATH:-}
SERVER_HOST="${SERVER_HOST:?e.g. SERVER_HOST=192.168.1.10 bash $0}"
python -m client.run_client \
  --host="$SERVER_HOST" \
  --port=8103 \
  --config_task_path=configs/task/cobot_screw_sorting.py
