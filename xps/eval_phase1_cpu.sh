#!/bin/bash
#SBATCH --job-name=eval_phase1_cpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-15
#SBATCH --output=logs/eval_phase1_%a.out

set -euo pipefail

module load eSpeak-NG
module load libsndfile
module load OpenBLAS

# --- user-configurable ---
LOGS_DIR="${LOGS_DIR:-/mnt/parscratch/users/acq22mc/exp/Mocha-TTS/logs/finetune_mocha}"
OUT_ROOT="${OUT_ROOT:-${LOGS_DIR}/eval_phase1}"
FILELIST="${FILELIST:?Set FILELIST to wav|text eval filelist path}"
STEPS="${STEPS:-10}"
TEMP="${TEMP:-0.667}"
LENGTH_SCALE="${LENGTH_SCALE:-1.0}"
F0_NJ="${F0_NJ:-8}"
MCD_NJ="${MCD_NJ:-8}"
MAX_UTTS="${MAX_UTTS:-}"
# ------------------------

source .venv/bin/activate

mapfile -t RUN_DIRS < <(find "${LOGS_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'phase1*' | sort)
TOTAL="${#RUN_DIRS[@]}"

if [[ "${TOTAL}" -eq 0 ]]; then
  echo "No phase1* run dirs found in ${LOGS_DIR}"
  exit 1
fi

IDX="${SLURM_ARRAY_TASK_ID}"
if [[ "${IDX}" -ge "${TOTAL}" ]]; then
  echo "SLURM_ARRAY_TASK_ID ${IDX} out of range (0..$((TOTAL - 1)))"
  exit 1
fi

RUN_DIR="${RUN_DIRS[$IDX]}"
RUN_NAME="$(basename "${RUN_DIR}")"

CKPT=""

# New layout: <run>/runs/<timestamp>/checkpoints/*.ckpt
latest_run="$(ls -1d "${RUN_DIR}"/runs/* 2>/dev/null | sort | tail -n 1 || true)"
if [[ -n "${latest_run}" ]]; then
  if [[ -f "${latest_run}/checkpoints/best.ckpt" ]]; then
    CKPT="${latest_run}/checkpoints/best.ckpt"
  elif [[ -f "${latest_run}/checkpoints/last.ckpt" ]]; then
    CKPT="${latest_run}/checkpoints/last.ckpt"
  else
    latest="$(ls -1 "${latest_run}"/checkpoints/checkpoint_*.ckpt 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "${latest}" ]]; then
      CKPT="${latest}"
    fi
  fi
fi

# Fallback older layouts
if [[ -z "${CKPT}" ]]; then
  if [[ -f "${RUN_DIR}/checkpoints/best.ckpt" ]]; then
    CKPT="${RUN_DIR}/checkpoints/best.ckpt"
  elif [[ -f "${RUN_DIR}/checkpoints/last.ckpt" ]]; then
    CKPT="${RUN_DIR}/checkpoints/last.ckpt"
  elif [[ -f "${RUN_DIR}/last.ckpt" ]]; then
    CKPT="${RUN_DIR}/last.ckpt"
  else
    latest="$(ls -1 "${RUN_DIR}"/checkpoints/checkpoint_*.ckpt 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "${latest}" ]]; then
      CKPT="${latest}"
    fi
  fi
fi

if [[ -z "${CKPT}" ]]; then
  echo "No checkpoint found for ${RUN_NAME} (${RUN_DIR})"
  exit 1
fi

OUTDIR="${OUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUTDIR}"

echo "Evaluating ${RUN_NAME}"
echo "Checkpoint: ${CKPT}"
echo "Outdir: ${OUTDIR}"

CMD=(python scripts/evaluate_checkpoint.py
  --checkpoint_path "${CKPT}"
  --filelist "${FILELIST}"
  --outdir "${OUTDIR}"
  --steps "${STEPS}"
  --temperature "${TEMP}"
  --length_scale "${LENGTH_SCALE}"
  --device cpu
  --f0_nj "${F0_NJ}"
  --mcd_nj "${MCD_NJ}"
)

if [[ -n "${MAX_UTTS}" ]]; then
  CMD+=(--max_utts "${MAX_UTTS}")
fi

"${CMD[@]}"
