#!/bin/bash
# A step on one GPU (04 smoke test, 08 comparison with v6, 09 final model, 10 predictions):
#     sbatch slurm/gpu_step.sbatch scripts/04_smoke_test.py configs/my_experiment.yaml
#     sbatch --time=2-00:00:00 slurm/gpu_step.sbatch scripts/09_train_final.py configs/my_experiment.yaml
#SBATCH --job-name=gouai_gpu
#SBATCH --partition=<add_partition>   # EDIT: your partition (or remove this line and export SBATCH_PARTITION=...)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
source slurm/_env.sh
SCRIPT=$1; CONFIG=$2; shift 2
"$PY" "$SCRIPT" --config "$CONFIG" "$@"
rc=$?; echo "finished at $(date) with exit code $rc"; exit $rc
