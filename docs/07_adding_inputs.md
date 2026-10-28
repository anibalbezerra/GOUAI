# 07 Adding inputs to the training

The v6 networks see two image channels of the entropy density (x/255 and x/S) and one scalar (S/max S).
This page shows how to give the network something else, from the simplest change (a configuration line)
to the most involved one (a new target). Every new input is a new **experiment**: give it its own
`run_dir`, run steps 01–08, and compare it with the baseline on the same test events
([05_evaluation_protocol.md](05_evaluation_protocol.md)). A network with new inputs cannot load the v6
weights (its input layer has a different shape), so it is always trained from scratch.

Where things happen:

| what | where | when |
|---|---|---|
| image channels | `gouai_train/features.py`: `SCALINGS`, `make_images` | built on the fly from `data/grid_*.npy` |
| scalar inputs | `features.py`: `scalar_raw`, normalized by `scalar_norms` | values from `data/event_table.csv` |
| per-event quantities computed from the grid | `features.py`: `DERIVED` | step 01 (stored as columns), and again when predicting |
| per-event quantities from outside | `data.extra_table`, `data.extra_y_columns` | step 01 (merged into the event table) |
| augmentation of a channel | the group (`abs` / `shape`) declared in `SCALINGS` | `pipeline.py` |
| network input shapes | `model.build_for_run`: number of channels and scalars from the config | automatic |

## Level 1: configuration only

**a) Energy instead of entropy density.** The paper's input comparison:

```yaml
input: energy
reference:
  old_data:
    x_file: /path/to/v6_data/2.76_PbPb/x_energy.dat
    x_kind: energy
    y_file: /path/to/v6_data/2.76_PbPb/Y_eta0p5_pt.dat
```
The channels become `energy_abs, energy_norm`, S becomes the total energy, the centrality table switches to
`<system>_energy.csv`, and step 08 compares with the published v6 *energy* networks. This needs
`data.x_kind: energy`.

**b) Both densities as channels** (4 channels):

```yaml
features:
  channels: [entropy_abs, entropy_norm, energy_abs, energy_norm]
```
The entropy and energy grids are both in `data/` whenever the raw files are energy density, so step 01
need not be rerun. Rerun 02 only for a new `run_dir` (or import the split), then 03 onwards.

**c) The eccentricities as extra scalars.** ε₂ and ε₃ of the entropy density, computed from each grid
about its centroid (`features.eccentricity`; it agrees with TRENTo's own ε₂, ε₃ to all printed digits):

```yaml
features:
  scalars: [S_norm, col:eps2_entropy, col:eps3_entropy]
```
Step 01 computes them and stores them as columns of `event_table.csv` (rerun step 01 in the new
`run_dir`). Step 10 recomputes them for new initial conditions automatically. Any quantity in
`features.DERIVED` can be used this way. `features.derived: [eps2_entropy]` stores a quantity without
using it as an input, e.g. for analysis.

**d) A per-event quantity from another source**, e.g. TRENTo's impact parameter b or N_part for each hybrid
event. Write a CSV with an `event_id` column (the same ids as the label file) and one column per quantity:

```
event_id,b,npart
0,7.96,225
1,14.14,23
```
```yaml
data:
  extra_table: /path/to/new_data/2.76_PbPb/trento_summary.csv
features:
  scalars: [S_norm, col:npart]
```
Events missing from the table are excluded (and listed in `excluded_events.csv`).
**Caution:** a network that needs `npart` can only predict events for which you know `npart`. For
TRENTo production (step 10 B) that is true for b and npart, which TRENTo prints, but not in general.
Columns of the label file can be added the same way with `data.extra_y_columns: {name: column}`. Never
use another final-state observable as an input to predict N_ch or [pT]: that would be circular.

## Level 2: a few lines of Python

**a) A new image channel**, e.g. log(1 + x), which compresses the dense core and enhances the dilute
edge. In `gouai_train/features.py`:

```python
SCALINGS = {
    'abs': (lambda x, S: x / 255.0, 'abs'),
    'norm': (lambda x, S: x / (S[:, None, None] + 1e-8), 'shape'),
    'log': (lambda x, S: np.log1p(x), 'abs'),          # new
}
```
then `features: channels: [entropy_abs, entropy_norm, entropy_log]`. The second element (`'abs'` or
`'shape'`) chooses the augmentation: `shape` channels get the stronger rotation and the small random shift
in the "extra" augmented copies. Keep channel values of order 0.01–10. Very large values slow training down.

**b) A new derived quantity**, e.g. the root-mean-square radius:

```python
def rms_radius(x):
    x = np.asarray(x, np.float64)
    c = (np.arange(x.shape[-1]) - (x.shape[-1] - 1) / 2) * GRID_STEP
    X, Y = np.meshgrid(c, c, indexing='ij')
    w = x.sum(axis=(1, 2))
    xc, yc = (x * X).sum(axis=(1, 2)) / w, (x * Y).sum(axis=(1, 2)) / w
    r2 = (X[None] - xc[:, None, None]) ** 2 + (Y[None] - yc[:, None, None]) ** 2
    return np.sqrt((x * r2).sum(axis=(1, 2)) / w)

DERIVED['rms_entropy'] = lambda grids: rms_radius(grids['entropy'])
```
then `scalars: [S_norm, col:rms_entropy]`.

**c) A new scalar computed from existing columns**, e.g. log S. In `scalar_raw` add
`if name == 'logS': return np.log(table[f'S_{input_density}'].values)`. Scalars are divided by their
maximum |value| over the training events, so keep them positive or centred sensibly.

After any code change: `python -m pytest tests -q`, then step 04 on a GPU. Its first check (the
untrained network reproduces g(S) exactly) still holds with any inputs, because the heads start at zero.

## Level 3: a new target (advanced)

Predicting a third observable touches several places, because N_ch and [pT] are written explicitly.
Examples are the further hydrodynamic moments in columns 3–10 of the label file (planned for future
GOUAI versions) or the elliptic flow v₂. Discuss the physics with your supervisor first, then:

1. **Label** (`rawdata.read_labels`, `data.y_columns`): read the new column. Check what each column of
   the label file is ([03_data_format.md](03_data_format.md) §1).
2. **Target transform** (`targets.py`): `TARGETS = ('nch', 'pt', 'v2')`, and `ResidualSpec.fit`,
   `transform` and `inverse` must take three arrays. The log-residual trend assumes a positive quantity
   that varies smoothly with S. That is fine for |v₂| or v₂{2}, but a signed quantity (e.g. a flow vector
   component) needs a linear residual instead, `d = y − g(S)`.
3. **Network** (`model.py`): add `'z_v2'` to the head loop, and a third Huber term in `make_loss`.
4. **Data** (`training.arrays`, `store.y`) and **outputs** (`training.prediction_table`,
   `ensemble._combine`): carry the third column.
5. **Metrics and reports** (`metrics.evaluate` loops over `TARGETS`; the report scripts and plots name
   `nch` and `pt` explicitly): add the new columns.
6. **Gate 2** (`06_cv_report.py`): add criteria for the new residual.

Expect v₂ to be much harder than N_ch: it depends weakly on S and strongly on fluctuations, and the
hybrid value of a single event carries finite-multiplicity noise.

## Checklist for any new input

- [ ] New `run_dir`; the same test set as the baseline (`splits.import_from: <baseline>/splits/splits.json`).
- [ ] `python -m pytest tests -q` passes; step 04 passes (zero-init check).
- [ ] The input is available for every event you want to predict later (step 10).
- [ ] Gate 2 passes; compare per class with the baseline using the per-event test predictions of both
      runs and `metrics.paired_bootstrap` ([06_code_reference.md](06_code_reference.md), metrics example).
- [ ] Report the result whether or not it helped.
