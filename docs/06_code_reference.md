# 06 Code reference

Every file, what its important parts do, and how it contributes to the whole. Read it next to the code:
each module also starts with a docstring that repeats the essentials.

```
            configs/*.yaml ──► config.py (Run)
                                   │
raw events ──► rawdata.py, eos.py ─┴─► 01 ──► data/  (event_table.csv, grid_*.npy)
                                          │
                       splits.py ──────► 02 ──► splits/splits.json
                                          │
             targets.py, features.py ──► 03 ──► targets/  (fold specs, gref, meta)
                                          │
  store.py, pipeline.py, model.py,        │
  attention.py, training.py ────────────► 04, 05, 09 ──► smoke/, cv/fold_k/, final/
                                          │
   metrics.py, centrality.py, plots.py ─► 06, 07 ──► cv/report.md, test/report.md
                                          │
                         ensemble.py ───► 08, 10 ──► compare_v6/, predict/
```

## 1. Library: `gouai_train/`

### `config.py`: configuration and the run directory

- **`load_config(path)`** reads `configs/defaults.yaml`, then merges the experiment file on top of it
  (`deep_merge`: nested keys are merged, so an experiment can change just `training.epochs`). Unknown
  top-level keys are rejected, which catches typos.
- **`resolve(cfg)`** fills defaults that depend on other keys (the channels from `input`, the centrality
  table from `system` and `input`, the EoS file, the v6 models folder), makes relative paths relative to the
  repository, and refuses a `run_dir` whose first directory is a symbolic link (`linked_top_dir`: it may
  point elsewhere on the compute nodes) or that still contains `CHANGE`.
- **`Run`** is what every script uses: `Run.from_args()` parses `--config` (plus script-specific
  options), creates `run_dir`, and offers `path()`, `subdir()`, `require()` (stops with "run step NN
  first" if an input is missing), `log()` (appends to `LOG.md` with the Slurm job id), `save_config()`
  (the merged configuration, for provenance), and JSON helpers that handle numpy types.
- `SYSTEMS` holds the labels of the two collision systems.

### `rawdata.py`: reading the hybrid events

- **`read_labels`** reads `y_file` (whitespace separated, `#` comments) and keeps the configured columns
  as `event_id`, `nch`, `pt` and any `extra_y_columns`. (`data.extra_table`, a CSV keyed by
  `event_id`, is merged in step 01.)
- **`match_report`** decides how grids and labels are paired: layout A by line number (it stops if the
  counts differ), layout B by the event id in the file name. It lists unmatched files and labels.
- **`iter_events`** yields `(label row, flat grid)` one event at a time, so memory stays small whatever
  the sample size. `read_grid_file` accepts one-column or 280 × 280 files.

### `eos.py`: energy → entropy density

`entropy_density(e, path)`: cubic interpolation of the HotQCD + HRG table s(e), and **s = 0 below the
smallest tabulated e**. This is exactly the conversion of the v6 training images (reproduces them to
3 × 10⁻⁷). Linear interpolation or extrapolation would change dilute regions by up to 10 %.

### `centrality.py`: minimum-bias centrality

`CentralityTable(path)(S)` interpolates the table `centrality_percent → S_min` (one row per 0.01 %) built
by `tools/make_centrality_table.py` from 1.5 million TRENTo events. `class_index(cent, edges)` assigns
classes (the last edge included). It is used only to sort test events into classes for reporting.

### `splits.py`: which events are used for what

- **`outer_split`**: the held-out test set, stratified on quantiles of S (`n_strata` bins), so every
  centrality is represented. Events in `forced_test` (the v6 held-out events found in the new sample)
  always go to the test set.
- **`make_folds`**: `StratifiedKFold` over the training events, also stratified on S. This is the same
  algorithm as for v6.
- **`subsample`**: a stratified subset of the training events (experiments E2/E3).
- **`check_split`**: asserts no overlaps, and that the validation sets cover the training events exactly once.
- **`match_events`**: finds identical events in two samples by (N_ch, [pT]) to 10⁻⁵, and raises if a match is ambiguous.

### `targets.py`: what the network learns (identical to v6)

- **`Trend`**: `log g(S) = P(log(S/S_ref))` with a cubic P fitted by least squares. Beyond the fitted
  range it continues linearly with the edge slope (`log_g`), so extrapolation never follows a cubic.
- **`ResidualSpec`**: one trend per target plus the mean μ and spread σ of the residual d.
  `transform(S, nch, pt) → z` (N, 2) and `inverse(S, z) → (nch, pt)`. Saved as `spec.json` next to every
  network, because each fold network has its own spec, fitted on its own training events.

### `features.py`: what the network sees (edit here to add inputs)

- **`SCALINGS`**: the ways a density grid becomes an image channel: `abs` (x/255) and `norm` (x/S), each
  with its augmentation group. Channel names are `<density>_<scaling>`.
- **`make_images(grids, sums, channels)`** builds the (N, 280, 280, C) float32 image in the configured
  channel order. The computation is done in float64, as for v6.
- **`scalar_raw` / `scalar_norms` / `make_scalars`**: scalar inputs (`S_norm`, `S_entropy`, `S_energy`,
  `col:<name>`), each divided by its maximum over the training events. The maxima are stored in
  `targets/meta.json` and reused for every prediction.
- **`DERIVED`**, **`eccentricity`**: per-event quantities computed from the grids (ε₂, ε₃ about the
  centroid; identical to TRENTo's own values). Step 01 stores them as columns, and
  `Ensemble.predict_grids` recomputes them for new initial conditions. `derived_needed(cfg)` lists the
  ones a run needs.

See [07_adding_inputs.md](07_adding_inputs.md).

### `store.py`: the prepared events of a run

`EventStore(run)` opens `event_table.csv` and memory-maps `grid_<density>.npy`, so opening is instant
and only the requested events are read. `images(idx)` and `scalars(idx, norms)` build network inputs
for any list of events (reading in increasing order, returning in the requested order).
`grids_sums(idx, densities)` returns raw grids, e.g. to feed the v6 networks, which have their own normalization.

### `pipeline.py`: input pipeline and augmentation (TensorFlow)

- **Augmentation**: `augment_mild` applies the same random flips and a rotation of up to ±18° to all
  channels. `make_augment_extra` transforms each channel independently: `abs` channels get mild
  transforms, `shape` channels get a rotation of up to ±54° plus a random shift of up to 14 pixels
  (reflect padding + random crop).
- **`make_dataset`**: for training, the dataset is a shuffled list of `(event, copy type)` pairs, with
  13 copies per event (1 plain, 8 mild, 4 extra). The image is fetched from memory (`tf.numpy_function`)
  and augmented on the fly. This reproduces the v6 epoch exactly (13 × N fully shuffled examples), while
  memory stays fixed for any N. v6 stored all 13 × N images in a shuffle buffer, which does not scale
  beyond ~1500 events. Validation data are served once, in order, unchanged.

### `attention.py`: spatial attention (TensorFlow, identical to v6)

`CBAMBlock` → `SpatialAttention`: the mean and max over channels → a 7 × 7 convolution → a sigmoid
weight per position, which multiplies the feature map. It must stay identical to load the v6 weights.

### `model.py`: the network (TensorFlow)

- **`build_model(tcfg, n_channels, n_scalars)`**: ResNet-50 (random init) → attention → dilated
  convolution → pooling → Dense 1024 → Dropout → Dense 128; the scalar branch is Dense 64-32-16; then a
  shared Dense 128 and two heads (Dense 32 + Dropout + Dense 1, L2-regularized, **zero-initialized**).
  With 2 channels and 1 scalar this is exactly the v6 network.
- **`build_for_run(cfg)`**: the same network with the run's channel and scalar counts.
- **`make_loss`**: Huber(z_Nch) + Huber(z_pT). **`compile_model`**: Adam with the configured learning rate.
- **`predict_z`**: raw network output z.

### `training.py`: training one network (TensorFlow)

- **`WarmupCheckpoint`**: saves the weights whenever the validation loss improves, but only after the
  warm-up. **`WarmupReduceLR`**: halves the learning rate after `rlr_patience` epochs without
  improvement, also only after the warm-up. `EarlyStopping(start_from_epoch=warmup)` stops after
  `es_patience` epochs without improvement.
- **`fit_network`**: seeds, builds datasets and network, trains, writes `history.csv`, reloads the best
  weights and copies `best.weights.h5` to the run directory. Checkpoints are written to node-local
  `/tmp` first (HDF5 locking on the network disk can hang). Without validation events (final model) it
  trains a fixed number of epochs.
- **`require_gpu`**: stops at once if TensorFlow sees no GPU.
- **`predict_physical` / `prediction_table`**: z → (N_ch, [pT]) through the network's spec, in chunks;
  one row per event with truth, prediction, z and S.

### `ensemble.py`: several networks together (TensorFlow)

`Ensemble.from_run(run)` loads the 5 fold networks of a run, `Ensemble.from_run(run, 'final')` the final
one, and `Ensemble.from_v6(models_dir, system, input)` the published networks. Every ensemble carries
its own channels and scalar normalization, so v6 is always fed its own max S. `predict_grids` /
`predict_store` return the geometric mean and the per-network log predictions (their spread is the
model uncertainty). Events with S = 0 get N_ch = 0 and [pT] = NaN.

### `metrics.py`: accuracy measures

- **`evaluate(S, y_true, y_pred, gref, cent, cfg)`** returns, for N_ch and [pT]: `all`, `nch_ge_floor`
  (N_ch ≥ 10), `classes` (per minimum-bias class) and `class_avg` (width-weighted 0–80 %). Each subset has
  `mae`, `mean_rel_err`, `bias` (physical; [pT] in MeV) and `r`, `beta_cal`, `err_ratio` (residual beyond
  g_ref, the same reference trend for every model).
- **`paired_bootstrap(err_a, err_b, n, seed)`**: the mean difference of two models' errors on the same
  events, with a 95 % interval.
- `flat_row`, `markdown_table`, `class_table`: the tables of the reports.

Example, a paired comparison of two of your own runs on their common test set (E1 vs E2):

```python
import pandas as pd
from gouai_train.metrics import paired_bootstrap
a = pd.read_csv('/path/to/gouai_runs/PbPb_new_entropy/test/predictions.csv').set_index('idx')
b = pd.read_csv('/path/to/gouai_runs/PbPb_new_entropy_n450/test/predictions.csv').set_index('idx')
b = b.loc[a.index]
central = a.centrality < 5
err = lambda d: (d.nch_pred / d.nch_true - 1).abs().values
print(paired_bootstrap(err(a[central]), err(b[central]), 2000, 0))   # diff < 0: E1 better
```

### `plots.py` and `style.py`: figures

One function per figure type (see the module docstring). `style.py` sets the common figure style
(boxed axes, inward ticks, bold labels), and `style.save` writes PNG and PDF.

### `gpu_env.py`

Preloads the CUDA libraries installed by pip before TensorFlow starts, so a GPU node with an incomplete
system CUDA does not make TensorFlow silently fall back to the CPU. It is imported automatically by the
TensorFlow modules.

## 2. Step scripts: `scripts/`

Every script has `--config` and a docstring (`python scripts/NN_*.py -h`). Each checks that its inputs
exist and appends one line to `LOG.md`.

| script | needs | writes | role |
|---|---|---|---|
| `00_check_environment.py` | – | – | packages, paths, write access, (GPU with `--gpu`) |
| `01_prepare_data.py` | raw files | `data/` | read, convert, check, find v6 events, centrality |
| `02_make_splits.py` | `data/` | `splits/` | held-out set + folds; refuses to overwrite |
| `03_fit_trends.py` | `splits/` | `targets/` | per-fold specs, g_ref, scalar normalization |
| `04_smoke_test.py` | `targets/` | `smoke/` | zero-init check + 2 epochs; time estimate |
| `05_train_fold.py --fold k` | `targets/` | `cv/fold_k/` | train network k; OOF + test predictions |
| `06_cv_report.py` | 5 folds | `cv/` | pooled OOF metrics, **Gate 2**, plan for 09 |
| `07_test_report.py` | Gate 2 passed | `test/` | ensemble on the held-out events |
| `08_compare_v6.py` | `test/`, v6 weights | `compare_v6/` | paired comparison with v6 (sets A and B) |
| `09_train_final.py` | `cv/final_plan.json` | `final/` | optional single network |
| `10_predict.py` | trained networks | `predict/` | ensemble on new initial conditions / TRENTo |
| `fetch_v6_weights.py` | – | `reference/v6/**/model.weights.h5` | links/copies + verifies the v6 weights |

## 3. Slurm scripts: `slurm/`

`_env.sh` (sourced by all) sets the environment: the Python of the active venv (or `$GOUAI_PYTHON`),
the CUDA library path, no HDF5 locking, thread counts.
`cpu_step.sbatch SCRIPT CONFIG`, `gpu_step.sbatch SCRIPT CONFIG`, `train_cv.sbatch CONFIG` (array 0–4)
and `predict_trento.sbatch CONFIG` (array over TRENTo tasks). Resources are in the `#SBATCH` lines at the top.

## 4. Reference data: `reference/`

- `centrality/<system>_<input>.csv`: minimum-bias centrality tables.
- `v6/<system>/splits.json`: the published held-out set and folds.
- `v6/<system>/<input>_pt/`: `ensemble.json` (training settings, max S), `fold_k/spec.json`,
  `fold_k/history.csv`, `fold_k/metrics.json`, `metrics.json`, `MODEL_CARD.md`, and predictions for
  every training event (`train_oof_predictions.csv`) and held-out event (`heldout_predictions.csv`).
  The weights `fold_k/model.weights.h5` come from `fetch_v6_weights.py`.

## 5. Tools: `tools/`

- `make_centrality_table.py`: builds a centrality table from any minimum-bias sample.
- `check_v6_reproduction.py`: proves that this code feeds the v6 networks as in training, and with
  `--install-as-cv` lets steps 06–08 run on the published networks (student guide Part 1).

## 6. Tests: `tests/`

`test_core.py` (targets, splits, centrality, features and eccentricities, EoS, raw-data layouts, config,
metrics) and
`test_network.py` (the 13-copy pipeline, zero-initialized output, input shapes). Run them after every
change to the code: `python -m pytest tests -q`.
