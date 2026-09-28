#!/usr/bin/env bash
# H20 RTC-SFT. Step 3 writes norm stats, step 4 runs train_offline_rtc.py.
# This is not OpenPI scripts/train.py.
# Checkpoints and norm stats stay in this repo. Only PI05_BASE is borrowed.
#   assets/pi05_cobot_screw_sorting/<repo_id>/norm_stats.json
#   checkpoints/cobot_screw_rtc_offline/rtc_screw_maxdelay10/
# Reattach after disconnect: tmux attach -t rtc_sft
set -euo pipefail

# SESSION="${TMUX_SESSION:-rtc_sft}"
# if [ -z "${TMUX:-}" ]; then
#   if tmux has-session -t "$SESSION" 2>/dev/null; then
#     echo "会话 $SESSION 已在运行。查看日志: tmux attach -t $SESSION"
#     exit 1
#   fi
#   script_path="$(readlink -f "$0")"
#   cd "$(dirname "$script_path")/../.."
#   tmux_env=()
#   for name in DO_STEP3 DO_STEP4 CUDA_VISIBLE_DEVICES repo_id config_name \
#     HF_LEROBOT_HOME HF_HOME HF_ENDPOINT PI05_BASE ASSETS_DIR ASSET_ID \
#     LEROBOT_ROOT CKPT_DIR XLA_PYTHON_CLIENT_MEM_FRACTION NCCL_NVLS_ENABLE; do
#     if [ -n "${!name+x}" ]; then
#       tmux_env+=(-e "$name=${!name}")
#     fi
#   done
#   echo "在 tmux 会话 $SESSION 中运行。断线后执行: tmux attach -t $SESSION"
#   exec tmux new-session -s "$SESSION" -c "$PWD" "${tmux_env[@]}" \
#     "exec bash $(printf '%q' "$script_path")"
# fi

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
source .venv/bin/activate

# 1 = run, 0 = skip
DO_STEP3="${DO_STEP3:-0}"  # norm stats
DO_STEP4="${DO_STEP4:-1}"  # RTC-SFT

export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export repo_id="${repo_id:-screw_sorting_single_sft_ep279_annotated}"
export config_name="${config_name:-pi05_cobot_screw_sorting}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/huggingface/lerobot}"
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
PI05_BASE="${PI05_BASE:-/workspace/openpi/models/checkpoints_pi05/pi05_base}"
ASSETS_DIR="${ASSETS_DIR:-$ROOT/assets/${config_name}}"
ASSET_ID="${ASSET_ID:-$repo_id}"
LEROBOT_ROOT="${LEROBOT_ROOT:-$HF_LEROBOT_HOME/$repo_id}"
CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints/cobot_screw_rtc_offline}"
NORM_STATS="$ASSETS_DIR/$ASSET_ID/norm_stats.json"

if [ "$DO_STEP3" -eq 1 ]; then
    echo "执行步骤3: 计算归一化统计量..."
    # CPU is enough. Keep JAX off the training GPUs while stats are computed.
    JAX_PLATFORMS=cpu python expo_ft/agents/vla/openpi/scripts/compute_norm_stats.py \
        --config-name "$config_name"
    echo "步骤3完成: $NORM_STATS"
else
    echo "跳过步骤3: 计算归一化统计量"
fi

if [ ! -f "$NORM_STATS" ]; then
    echo "缺少归一化统计: $NORM_STATS" >&2
    exit 1
fi

if [ "$DO_STEP4" -eq 1 ]; then
    echo "执行步骤4: RTC-SFT..."
    python train_offline_rtc.py \
        --config_task=configs/task/cobot_screw_sorting.py \
        --dataset_path="$LEROBOT_ROOT" \
        --num_data=0 \
        --batch_size=64 \
        --fsdp_devices=1 \
        --replan_steps=8 \
        --config=configs/model/rtc_pi_config.py \
        --config.freeze_pi05_encoder=False \
        --config.p1_use_prefix_conditioning=True \
        --config.p1_max_delay=10 \
        --config.pi05_config_name="$config_name" \
        --config.pi05_weight_loader_path="$PI05_BASE/params" \
        --config.pi05_assets_dir="$ASSETS_DIR" \
        --config.pi05_asset_id="$ASSET_ID" \
        --project_name=rtc_sft_cobot_screw \
        --output_dir="$CKPT_DIR" \
        --max_steps=4000 \
        --resume \
        --checkpoint_model \
        --checkpoint_interval=2000 \
        --run_name=rtc_screw_maxdelay10
    echo "步骤4完成"
else
    echo "跳过步骤4: RTC-SFT"
fi
