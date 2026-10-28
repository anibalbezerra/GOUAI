#!/bin/bash
# STEP 05: the 5 cross-validation networks, one GPU each, all at the same time (Slurm array 0-4):
#     sbatch slurm/train_cv.sbatch configs/my_experiment.yaml
# Retrain only some folds (e.g. 1 and 3 failed):
#     sbatch --array=1,3 slurm/train_cv.sbatch configs/my_experiment.yaml
# Time: ~1-1.5 h per fold for ~450 training events; it grows about linearly with the number of
# events (step 04 prints an estimate). The limit below is generous on purpose.
#SBATCH --job-name=gouai_cv
#SBATCH --partition=<add_partition>   # EDIT: your partition (or remove this line and export SBATCH_PARTITION=...)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -uo pipefail
source slurm/_env.sh
CONFIG=$1
"$PY" scripts/05_train_fold.py --config "$CONFIG" --fold "$SLURM_ARRAY_TASK_ID"
rc=$?; echo "finished at $(date) with exit code $rc"; exit $rc
