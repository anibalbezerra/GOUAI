# 02 Installation and cluster rules

## 1. Where to put things

Clone the repository, keep your data and run directories on a disk that the **compute nodes** see under
the **same path** as the login node.

**Watch out for symbolic links.** On some clusters a short path on the login node (say `/scratch`) is a
symbolic link to the shared disk (say `/home/scratch`), while on the compute nodes the same short path is
a different, node-local disk. A job that refers to the short path then fails within seconds without a
log, or cannot find your files. Always use the resolved path. The code refuses a `run_dir` whose first
directory is a symbolic link, and `00_check_environment.py` checks your working directory. Your
supervisor's lab notes give the right paths for our cluster.

```bash
cd <shared disk>/$USER
git clone https://github.com/anibalbezerra/GOUAI-train.git
cd GOUAI-train                         # always submit jobs from here
```

Keep run directories outside the repository, e.g. `<shared disk>/$USER/gouai_runs/<experiment>`.
A run with 3000 events needs about 1–2 GB for the prepared data plus 135 MB per trained network.

## 2. Python environment

**Option A, a shared lab environment**, if your group has one (path in the lab notes). The code was
tested with Python 3.13, TensorFlow 2.21 / Keras 3.13 and bundled CUDA.

**Option B, your own environment** (any Linux machine with an NVIDIA GPU):

```bash
python3 -m venv <shared disk>/$USER/venvs/gouai
source <shared disk>/$USER/venvs/gouai/bin/activate
pip install --upgrade pip
pip install -r requirements.txt        # tensorflow[and-cuda], numpy, pandas, scipy, scikit-learn, matplotlib, pyyaml, h5py, pytest
```

On a laptop without an NVIDIA GPU, install `tensorflow` instead of `tensorflow[and-cuda]`. Steps 01–03,
06 and 07 and the tests run fine on a CPU; training does not (it is ~100 times slower).

**How jobs find Python.** The job scripts use the `python` of the environment that is active when you run
`sbatch`, because Slurm passes your environment on to the job. So activate the environment first,
or set `export GOUAI_PYTHON=/path/to/venv/bin/python`.

## 3. Check the installation

```bash
python -m pytest tests -q                                       # ~1 minute: 17 passed
```

Then make your first configuration (student guide, Part 1), e.g. `configs/repro_PbPb.yaml` copied from
`configs/examples/reproduce_v6_PbPb.yaml` with its `run_dir` edited. Check it on the login node and,
inside a short GPU job, check that TensorFlow sees a GPU:

```bash
python scripts/00_check_environment.py --config configs/repro_PbPb.yaml
srun --partition=<gpu partition> --gres=gpu:1 --time=00:10:00 --pty \
     python scripts/00_check_environment.py --config configs/repro_PbPb.yaml --gpu
```

(The example files themselves are refused until their `CHANGE` lines are edited.)

The last line must read `GPU visible to TensorFlow: [PhysicalDevice(name='/physical_device:GPU:0' ...`.

## 4. Cluster conventions

| | |
|---|---|
| partition | set it once in every `slurm/*.sbatch` (the `<add_partition>` line), or delete that line and `export SBATCH_PARTITION=<name>` |
| nodes to avoid | `export SBATCH_EXCLUDE=<node list>` (Slurm reads it for every `sbatch`), as your lab notes say |
| GPU | one network uses one GPU; developed on RTX 6000 Ada (48 GB) |
| CPU steps | quick ones (02, 03, 06, 07) run directly on the login node; step 01 with thousands of events can go to a job (`slurm/cpu_step.sbatch`) |
| logs | `logs/<job-name>_<jobid>.out/.err` in the repository; the summary of every step in `<run_dir>/LOG.md` |
| monitoring | `squeue -u $USER`, `sacct -j <jobid>`, `tail -f logs/gouai_cv_<jobid>_0.out` |
| cancelling | `scancel <jobid>` (only your own jobs) |

**Before submitting long jobs, tell your supervisor what you will submit and for how long.** The GPUs are shared.

## 5. The published v6 weights (needed for step 08 only)

The 20 weight files (135 MB each) are not in git. If you have a copy (inside the lab: the folder in the
lab notes), run

```bash
python scripts/fetch_v6_weights.py --from-dir <folder with the weight files>
```

This creates symbolic links (`--copy` to copy instead) and verifies every file's checksum. Otherwise
download them with `--zenodo <record id>` from the record of the GOUAI release.
