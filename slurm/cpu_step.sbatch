#!/bin/bash
# A step that needs no GPU (01, 02, 03, 06, 07), as a batch job:
#     sbatch slurm/cpu_step.sbatch scripts/01_prepare_data.py configs/my_experiment.yaml
# Steps 02, 03, 06 and 07 take seconds: you may also run them directly on the login node.
#SBATCH --job-name=gouai_cpu
#SBATCH --partition=<add_partition>          # any partition; these steps need no GPU
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
source slurm/_env.sh
SCRIPT=$1; CONFIG=$2; shift 2
"$PY" "$SCRIPT" --config "$CONFIG" "$@"
rc=$?; echo "finished at $(date) with exit code $rc"; exit $rc
