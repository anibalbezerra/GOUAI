# 08 Troubleshooting

First look at three places: the last lines of `<run_dir>/LOG.md`, the job's `logs/*.err` and
`logs/*.out`, and `sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,NodeList`.

## Jobs

| symptom | cause | fix |
|---|---|---|
| job ends within a minute, state FAILED, no log file | submitted from a path the compute node does not have (e.g. a symbolic link) | `cd` to the resolved path (`00_check_environment.py` tells you) and resubmit |
| `sbatch: error: invalid partition specified: <add_partition>` | partition not set | edit the `--partition` line of the sbatch script, or `export SBATCH_PARTITION=<name>` |
| `sbatch: error: ... logs/...: No such file or directory` | `logs/` missing | `mkdir logs` in the repository |
| `python: command not found` / `No module named tensorflow` in the log | no environment active when you ran `sbatch` | activate the venv, then `sbatch`; or `export GOUAI_PYTHON=/path/to/venv/bin/python` |
| `NO GPU visible ... stopped` | the job got no GPU, or TensorFlow cannot load CUDA on that node | check `--gres=gpu:1` in the script; check the node with `00_check_environment.py --gpu`; report the node name to your supervisor and add `--exclude=<node>` |
| `Cannot dlopen some GPU libraries` | CUDA libraries missing on the node | normally handled by `gpu_env.py`; if it persists, as above |
| job PENDING for a long time | GPUs busy | `squeue -p <partition>`; wait, do not resubmit copies |
| state TIMEOUT in step 05 | fold slower than the time limit | `sbatch --time=3-00:00:00 --array=<k> slurm/train_cv.sbatch $CFG`; check the step-04 estimate |
| state OUT_OF_MEMORY | very large sample in memory | `--mem=128G` in the sbatch script |
| a fold job hangs with no output for hours | HDF5 locking on the network disk | `_env.sh` sets `HDF5_USE_FILE_LOCKING=FALSE`; make sure you use the provided job scripts |

## Steps

| message | meaning | fix |
|---|---|---|
| `CONFIG ERROR: unknown top-level keys` | typo in a key of your YAML | compare with `configs/defaults.yaml` |
| `CONFIG ERROR: run_dir ...: /x is a symbolic link` | see above | use the resolved path it suggests |
| `CONFIG ERROR: ... still contain CHANGE` | example config not edited | edit every line marked CHANGE (or delete `reference.old_data`) |
| `MISSING: ... Run step NN first` | steps run out of order | run the step named |
| step 01: `has N lines but ... has M rows` | layout A with a missing/extra event | use layout B (one file per event, matched by id), or fix the files |
| step 01: many `no grid file` events | ids in file names ≠ ids in the label file | check `data.x_pattern` and `data.y_columns.id` |
| step 01: WARNING `[pT] outside 0.2-2 GeV` | wrong column or unit (MeV?) | check `data.y_columns`; [pT] must be in GeV |
| step 02: `splits.json exists` | protection against changing the test set | only if no network was trained: delete it, or `--force` |
| step 05: `training ended before the warm-up was over` | `training.epochs` ≤ `warmup_epochs` | do not lower `epochs` below ~300 |
| step 07: `Gate 2 did not pass` | by design | see below |
| step 08: `model.weights.h5 missing` | v6 weights not fetched | `python scripts/fetch_v6_weights.py --from-dir ...` |
| step 08: `WARNING: v6 predictions differ from the published ones` | wrong `reference.old_data` file (entropy vs energy?) or broken weights | check the paths and `x_kind`; rerun `fetch_v6_weights.py` (it verifies checksums) |
| `tf.function retracing` warnings in step 08/10 | each of the 5 networks compiles its own predict function | harmless |

## Gate 2 fails

Do not look at the test set. Read `cv/report.md`: which criterion failed, and in which fold?

- **(c) convergence in one fold** (best/epoch-1 > 0.5, or no early stop before 1000 epochs). Open
  `cv/learning_curves.png`. If one fold's validation loss never settled, retrain that fold with another
  seed (`training.seed` changes all folds; to change one fold only, ask your supervisor) and record it.
- **(a)/(b) err/σ ≥ 1 or β small in one fold**: that fold's network did not learn the residual. Check its
  validation events (`cv/fold_k/predictions.csv`, part `oof`): a cluster of bad events may be bad *data*
  (wrong labels, mismatched grids). Go back to step 01's report.
- **Fails everywhere**: most likely a data problem (labels misaligned with grids). Run the student-guide
  Part 1 reproduction to check your environment. Then compare N_ch vs S in `data_overview.png`: a cloud
  instead of a tight band means the grids and labels are paired wrongly.

Any rerun after a failure goes into the report ([05_evaluation_protocol.md](05_evaluation_protocol.md) §1).

## Results look strange

- **Large errors only in 90–100 %**: expected. N_ch is a few particles there; v6 has 20–30 % errors in
  that class.
- **New networks worse than v6 on set B (v6 test events)**: possible if the new sample has fewer events
  in the classes that dominate set B. Compare the step-01 class tables of the two samples.
- **Headline MAE larger than the paper's**: probably more central events in your test set (larger N_ch,
  larger absolute errors). Compare class-averaged relative errors instead ([05](05_evaluation_protocol.md) §2).
