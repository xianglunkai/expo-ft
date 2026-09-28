#!/usr/bin/env bash
# Eval the H20 RTC-SFT checkpoint alone on the Cobot.
# RTCLearner: pi0.5 with prefix inpainting. No edit actor, no Q selection, no online checkpoint.
# IPC: SERVER_HOST=<this machine> bash scripts/cobot_screw_sorting/run_policy.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

export config_name="${config_name:-pi05_cobot_screw_sorting}"
LEROBOT_ROOT="${LEROBOT_ROOT:-/data/huggingface/lerobot/openpi/screw_sorting_single_sft}"
FROM_H20=./checkpoints/from_h20
RTC_PARAMS="${RTC_PARAMS:-$FROM_H20/rtc_screw_maxdelay10/checkpoints/4000/params}"
ASSETS_DIR="${ASSETS_DIR:-$FROM_H20/assets/${config_name}}"
ASSET_ID="${ASSET_ID:-screw_sorting_single_sft_ep279_annotated}"
NUM_EPISODES="${NUM_EPISODES:-30}"

if [[ ! -d "$RTC_PARAMS" ]]; then
  echo "Missing H20 checkpoint: $RTC_PARAMS" >&2
  echo "Download rtc_screw_maxdelay10 into checkpoints/from_h20/ first." >&2
  exit 1
fi

python eval_droid_policy.py \
  --config_task=configs/task/cobot_screw_sorting.py \
  --config=configs/model/rtc_pi_config.py \
  --dataset_path="$LEROBOT_ROOT" --num_data=1 \
  --client_host=0.0.0.0 --client_port=8103 \
  --config.pi05_config_name="$config_name" \
  --config.pi05_weight_loader_path="$RTC_PARAMS" \
  --config.pi05_assets_dir="$ASSETS_DIR" \
  --config.pi05_asset_id="$ASSET_ID" \
  --delay=5 --replan_steps=8 --num_episodes="$NUM_EPISODES"
