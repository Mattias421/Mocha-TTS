#!/bin/bash
#SBATCH --job-name=mocha_phase1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/mocha_phase1_%a.out
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-15

set -euo pipefail

module load eSpeak-NG/1.51-gfbf-2023a
ml CUDA/12.4.0
ml SoX/14.4.2-GCCcore-12.3.0

source .venv/bin/activate

# ---- Phase 1 grid (16 runs) ----
# freeze_decoder: true,false
# lr: 1e-4,2e-4
# dt: 1.0,0.5
# time_norm_mode: utterance,global
# fixed: hidden_channels=64, interpolation=linear, time_norm_value=1024

FREEZE=(true false)
LR=(1e-4 2e-4)
DT=(1.0 0.5)
TIME_NORM=(utterance global)

IDX=${SLURM_ARRAY_TASK_ID}

N_FREEZE=${#FREEZE[@]}
N_LR=${#LR[@]}
N_DT=${#DT[@]}
N_TIME=${#TIME_NORM[@]}
TOTAL=$((N_FREEZE * N_LR * N_DT * N_TIME))

if [[ ${IDX} -ge ${TOTAL} ]]; then
  echo "SLURM_ARRAY_TASK_ID ${IDX} out of range (0..$((TOTAL - 1)))"
  exit 1
fi

freeze=${FREEZE[$((IDX % N_FREEZE))]}
IDX=$((IDX / N_FREEZE))
lr=${LR[$((IDX % N_LR))]}
IDX=$((IDX / N_LR))
dt=${DT[$((IDX % N_DT))]}
IDX=$((IDX / N_DT))
time_norm=${TIME_NORM[$((IDX % N_TIME))]}

run_name="phase1_lr${lr}_dt${dt}_frz${freeze}_tn${time_norm}_id${SLURM_ARRAY_TASK_ID}"

echo "Running ${run_name}"
echo "freeze_decoder=${freeze} lr=${lr} dt=${dt} time_norm_mode=${time_norm}"

export WANDB_TAGS="mocha_phase1,lr_${lr},dt_${dt},freeze_${freeze},tn_${time_norm}"

python matcha/finetune_mocha.py \
  run_name="${run_name}" \
  model.cde.enabled=true \
  freeze_decoder="${freeze}" \
  model.optimizer.lr="${lr}" \
  data.batch_size=64 \
  model.cde.hidden_channels=64 \
  model.cde.interpolation=linear \
  model.cde.dt="${dt}" \
  model.cde.time_norm_mode="${time_norm}" \
  model.cde.time_norm_value=1024 \
  trainer.max_steps=101 \
  +trainer.limit_val_batches=8
