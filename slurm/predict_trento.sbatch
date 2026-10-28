#!/bin/bash
# STEP 10 B: TRENTo + network for production.n_events events, split over production.n_tasks GPU tasks:
#     sbatch --array=0-7 slurm/predict_trento.sbatch configs/my_experiment.yaml     (array = 0 .. n_tasks-1)
# then, when all tasks are done:  python scripts/10_predict.py --config ... --trento-merge
# Needs the trento binary on PATH (or production.trento_bin). Killed tasks resume: resubmit them.
#SBATCH --job-name=gouai_trento
#SBATCH --partition=<add_partition>   # EDIT: your partition (or remove this line and export SBATCH_PARTITION=...)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -uo pipefail
source slurm/_env.sh
CONFIG=$1
"$PY" scripts/10_predict.py --config "$CONFIG" --trento-task "$SLURM_ARRAY_TASK_ID"
rc=$?; echo "finished at $(date) with exit code $rc"; exit $rc
