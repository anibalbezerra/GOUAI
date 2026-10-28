# Sourced by every job script: the software environment of the jobs.
# Python: the one you activated before calling sbatch (sbatch passes your environment on),
# or set GOUAI_PYTHON=/path/to/venv/bin/python.
PY=${GOUAI_PYTHON:-python}

export HDF5_USE_FILE_LOCKING=FALSE        # HDF5 locking on the network disk can hang the job
export TF_CPP_MIN_LOG_LEVEL=1             # fewer TensorFlow messages
export TF_GPU_ALLOCATOR=cuda_malloc_async
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TF_NUM_INTEROP_THREADS=1
export MPLBACKEND=Agg
# CUDA libraries installed by pip (tensorflow[and-cuda]) on the loader path
NV=$("$PY" -c 'import site,os; print(next((os.path.join(p,"nvidia") for p in site.getsitepackages() if os.path.isdir(os.path.join(p,"nvidia"))), ""))' 2>/dev/null)
[ -n "$NV" ] && export LD_LIBRARY_PATH=$(ls -d "$NV"/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}

echo "job ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID} on $(hostname) at $(date); python: $PY"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
