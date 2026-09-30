#!/usr/bin/env bash
# Local 3090 online RL. RTC-SFT is trained on H20; download it first:
#   rsync -aP <h20>:.../checkpoints/cobot_screw_rtc_offline/rtc_screw_maxdelay10/ \
#     checkpoints/from_h20/rtc_screw_maxdelay10/
#   rsync -aP <h20>:<expo-ft>/assets/pi05_cobot_screw_sorting/ \
#     checkpoints/from_h20/assets/pi05_cobot_screw_sorting/
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

export config_name="${config_name:-pi05_cobot_screw_sorting}"
LEROBOT_ROOT="${LEROBOT_ROOT:-/data/huggingface/lerobot/openpi/screw_sorting_single_sft}"
FROM_H20=./checkpoints/from_h20
RTC_PARAMS="${RTC_PARAMS:-$FROM_H20/rtc_screw_maxdelay10/checkpoints/4000/params}"
ASSETS_DIR="${ASSETS_DIR:-$FROM_H20/assets/${config_name}}"
ASSET_ID="${ASSET_ID:-screw_sorting_single_sft_ep279_annotated}"

if [[ ! -d "$RTC_PARAMS" ]]; then
  echo "Missing H20 checkpoint: $RTC_PARAMS" >&2
  echo "Download rtc_screw_maxdelay10 into checkpoints/from_h20/ first." >&2
  exit 1
fi

python train_pi_robo.py \
  --config_task=configs/task/cobot_screw_sorting.py \
  --dataset_path="$LEROBOT_ROOT" \
  --num_data=0 \
  --batch_size=8 \
  --update_type=episode \
  --step_interval=30 \
  --offline_ratio=0 \
  --max_episodes="${MAX_EPISODES:-50}" \
  --config=configs/model/realtime_expo_ft_pi_config.py \
  --config.train_base_actor=False \
  --config.valids_keep_terminal_windows=True \
  --config.N=8 \
  --config.n_edit_samples=8 \
  --config.filter_N=8  \
  --config.filter_n_edit=1 \
  --config.edit_scale=0.1 \
  --config.filter_add_delayed_obs=True \
  --config.pi05_config_name="$config_name" \
  --config.pi05_weight_loader_path="$RTC_PARAMS" \
  --config.pi05_assets_dir="$ASSETS_DIR" \
  --config.pi05_asset_id="$ASSET_ID" \
  --project_name=rt_expo_cobot_screw \
  --output_dir=./checkpoints/cobot_screw_online \
  --client_host=0.0.0.0 --client_port=8103 \
  --fsdp_devices=1 \
  --delay=6 --replan_steps=10 \
  --resume \
  --checkpoint_model \
  --checkpoint_buffer \
  --checkpoint_interval=2000 \
  --run_name=rt_screw_single_delay5
