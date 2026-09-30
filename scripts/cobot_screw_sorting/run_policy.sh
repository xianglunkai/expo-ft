#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source /opt/ros/noetic/setup.bash
source client/.venv/bin/activate
# AgileX real_env lives under openpi; do NOT prepend /usr/lib/python3/dist-packages
# (system numpy 1.17 breaks the 3.11 venv). rospkg/dm_env are installed in client/.venv.
export PYTHONPATH="/home/xlk/work/openpi${PYTHONPATH:+:$PYTHONPATH}"
# Same machine as eval_rtc_sft / run_server: 127.0.0.1. Other host: SERVER_HOST=<ip> bash $0
export SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
python -m client.run_client \
  --host="$SERVER_HOST" \
  --port=8103 \
  --config_task_path=configs/task/cobot_screw_sorting.py
