#!/bin/bash
#SBATCH --job-name=mocha_phase2
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/mocha_phase2_%A_%a.out
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-0

set -euo pipefail

ml eSpeak-NG/1.50-gompi-2020a
ml CUDA/12.4.0
source .venv/bin/activate

# -------- TEMPLATE --------
# Fill this with top configs from phase 1.
# Example refinement dimensions:
# - LR around winner: e.g. 1.5e-4, 2e-4, 2.5e-4
# - depth: 4,6
# - hidden_channels: 64,96
# - seed: 1234,4321

LR=(2e-4)
DEPTH=(4)
HIDDEN=(64)
SEED=(1234)
FREEZE=(true)
DT=(0.5)
TIME_NORM=(utterance)

IDX=${SLURM_ARRAY_TASK_ID}
N_LR=${#LR[@]}
N_DEPTH=${#DEPTH[@]}
N_HIDDEN=${#HIDDEN[@]}
N_SEED=${#SEED[@]}
N_FREEZE=${#FREEZE[@]}
N_DT=${#DT[@]}
N_TIME=${#TIME_NORM[@]}
TOTAL=$((N_LR * N_DEPTH * N_HIDDEN * N_SEED * N_FREEZE * N_DT * N_TIME))

if [[ ${IDX} -ge ${TOTAL} ]]; then
  echo "SLURM_ARRAY_TASK_ID ${IDX} out of range (0..$((TOTAL - 1)))"
  exit 1
fi

lr=${LR[$((IDX % N_LR))]}
IDX=$((IDX / N_LR))
depth=${DEPTH[$((IDX % N_DEPTH))]}
IDX=$((IDX / N_DEPTH))
hidden=${HIDDEN[$((IDX % N_HIDDEN))]}
IDX=$((IDX / N_HIDDEN))
seed=${SEED[$((IDX % N_SEED))]}
IDX=$((IDX / N_SEED))
freeze=${FREEZE[$((IDX % N_FREEZE))]}
IDX=$((IDX / N_FREEZE))
dt=${DT[$((IDX % N_DT))]}
IDX=$((IDX / N_DT))
time_norm=${TIME_NORM[$((IDX % N_TIME))]}

run_name="phase2_lr${lr}_d${depth}_h${hidden}_s${seed}_frz${freeze}_dt${dt}_tn${time_norm}_id${SLURM_ARRAY_TASK_ID}"

echo "Running ${run_name}"

export WANDB_TAGS="mocha_phase2,lr_${lr},depth_${depth},hidden_${hidden},seed_${seed},freeze_${freeze},dt_${dt},tn_${time_norm}"

python matcha/finetune_mocha.py \
  run_name="${run_name}" \
  seed="${seed}" \
  model.cde.enabled=true \
  freeze_decoder="${freeze}" \
  model.optimizer.lr="${lr}" \
  data.batch_size=64 \
  model.cde.num_layers="${depth}" \
  model.cde.hidden_channels="${hidden}" \
  model.cde.interpolation=linear \
  model.cde.dt="${dt}" \
  model.cde.time_norm_mode="${time_norm}" \
  model.cde.time_norm_value=1024 \
  +trainer.max_steps=20000 \
  +trainer.limit_val_batches=16
