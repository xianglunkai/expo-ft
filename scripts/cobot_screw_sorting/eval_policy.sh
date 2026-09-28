#!/usr/bin/env bash
# Local eval of the online RL policy. Same downloaded H20 RTC weights and norm stats as run_server.sh.
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

if [[ ! -d "$RTC_PARAMS" ]]; then
  echo "Missing H20 checkpoint: $RTC_PARAMS" >&2
  exit 1
fi

python eval_droid_policy.py \
  --config_task=configs/task/cobot_screw_sorting.py \
  --config=configs/model/realtime_expo_ft_pi_config.py \
  --dataset_path="$LEROBOT_ROOT" --num_data=1 \
  --client_host=0.0.0.0 --client_port=8103 \
  --config.N=8 --config.n_edit_samples=8 \
  --config.filter_N=8 --config.filter_n_edit=1 \
  --config.edit_scale=0.1 --config.filter_add_delayed_obs=True \
  --config.pi05_config_name="$config_name" \
  --config.pi05_weight_loader_path="$RTC_PARAMS" \
  --config.pi05_assets_dir="$ASSETS_DIR" \
  --config.pi05_asset_id="$ASSET_ID" \
  --checkpoint_dir=./checkpoints/cobot_screw_online/rt_screw_single_delay5/checkpoints \
  --delay=5 --replan_steps=8 --num_episodes=30
