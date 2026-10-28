# 04 Student guide: the procedure, step by step

This is the procedure you will follow. Each step is one command that you start yourself. After each
step, **look at what it wrote** (the ✅ boxes say what to check) before going on. Nothing is automatic:
if something looks wrong, stop and ask.

One student works on **Au–Au 200 GeV** (`0.2_AuAu`), the other on **Pb–Pb 2.76 TeV** (`2.76_PbPb`).
The procedure is identical. Only the configuration file differs.

Keep a short lab notebook (a text file is enough): date, step, job id, what you saw, anything odd.
[`docs/results_template.md`](results_template.md) is the form in which you will report the results.

---

## Part 0: one-time setup (30 minutes)

1. Read [01_background.md](01_background.md).
2. Install as in [02_installation.md](02_installation.md): clone on the shared disk, activate
   the environment, `python -m pytest tests -q` → `17 passed`.
3. Fetch the published v6 weights (needed in Part 1 and in step 08):
   ```bash
   python scripts/fetch_v6_weights.py --from-dir <weights folder from the lab notes>
   ```
   ✅ `20 weight files in place and verified`.

---

## Part 1: warm-up, run the v6 procedure on the original data (1 hour, no GPU)

Before using new data, run the analysis steps on the **original** v6 events with the **published**
networks. You learn every step and every output on a case where the right answer is known.
No training is involved: the published networks stand in for step 05.

```bash
cp configs/examples/reproduce_v6_AuAu.yaml configs/repro_AuAu.yaml     # Pb-Pb student: reproduce_v6_PbPb.yaml
#   edit the CHANGE lines: run_dir (your folder) and the v6 data paths (lab notes)
CFG=configs/repro_AuAu.yaml
python scripts/00_check_environment.py --config $CFG
python scripts/01_prepare_data.py      --config $CFG
python scripts/02_make_splits.py       --config $CFG
python scripts/03_fit_trends.py        --config $CFG
python tools/check_v6_reproduction.py  --config $CFG --install-as-cv    # ~2-5 min on the login node
python scripts/06_cv_report.py         --config $CFG
python scripts/07_test_report.py       --config $CFG
python scripts/08_compare_v6.py        --config $CFG                    # ~3-5 min on the login node
```

✅ What you must get:
- `check_v6_reproduction`: `OK ... to ~1e-4` (the published numbers were computed on GPUs; see the tool's docstring).
- `cv/report.md`: **Gate 2: PASS**.
- `test/report.md`: N_ch MAE **2.82** and [pT] MAE **5.08 MeV** for Au–Au (**4.76** and **5.53 MeV** for Pb–Pb): the paper's numbers.
- `compare_v6/report.md`: differences of 0.00 %, because here "new" and "v6" are the same networks.

Open every report and figure, and make sure you understand each table using [01_background.md](01_background.md)
and the [glossary](09_glossary.md). Discuss them with your supervisor before Part 2.

*Optional, recommended once the GPUs are free:* retrain this reproduction run for real (steps 04 and 05
below, then 06–08) in a **new** run directory. Your Gate 2 and test numbers should agree with the
published ones within the run-to-run scatter of GPU training (a few to ~10 % on the errors). This shows
that your environment trains correctly.

---

## Part 2: your new sample

### Configuration

```bash
cp configs/examples/AuAu_new_entropy.yaml configs/AuAu_new.yaml        # Pb-Pb: PbPb_new_entropy.yaml
```

Edit the lines marked `CHANGE`:
- `run_dir`: a new directory for this experiment, e.g. `<shared disk>/<you>/gouai_runs/AuAu_new_entropy`;
- `data.x_path`: the directory with one energy-density file per event (or a single file, one event per line);
- `data.y_file`: the label file;
- `reference.old_data`: the v6 data files (paths in the lab notes).

Do not change anything else. All other settings come from `configs/defaults.yaml` and are those of v6.
Below, `CFG=configs/AuAu_new.yaml`.

### Step 00: check (seconds, login node)

```bash
python scripts/00_check_environment.py --config $CFG
```
✅ `ALL CHECKS PASSED`. A WARN about `v6 weights` means Part 0.3 was skipped.
❌ FAIL: read the message; most are wrong paths ([08_troubleshooting.md](08_troubleshooting.md)).

### Step 01: prepare the data (minutes, login node or `sbatch slurm/cpu_step.sbatch scripts/01_prepare_data.py $CFG`)

```bash
python scripts/01_prepare_data.py --config $CFG
```
Reads every event, converts energy to entropy density, checks the events, and finds the events that are
also in the v6 sample. See [03_data_format.md](03_data_format.md).

✅ Read `<run_dir>/data/data_report.md` and look at `data_overview.png`:
- the number of usable events is what you expected; excluded events (`excluded_events.csv`) are few and explained;
- no WARNING lines (or you understand them);
- the table "events per centrality class" shows the new sample's extra ultra-central events (0–1 %, 1–2 %, 2–5 %) next to v6's;
- panel (b): N_ch rises smoothly with S over ~3 decades; panel (c): [pT] between ~0.45 and ~0.75 GeV.

Write the numbers of the centrality table in your notebook. They are the "input" of the whole project.

### Step 02: the held-out test set and the folds (seconds, login node)

```bash
python scripts/02_make_splits.py --config $CFG
```
✅ `splits/splits_report.md`: about 25 % of the events in the test set, **every centrality class
present in both train and test**, and the note `N v6 held-out events forced into the test set` (if the
new sample contains v6 events).

From now on the split must not change. The script refuses to overwrite it. **Never look at test-set
results to tune anything** ([05_evaluation_protocol.md](05_evaluation_protocol.md) §1).

### Step 03: trend and normalizations (seconds, login node)

```bash
python scripts/03_fit_trends.py --config $CFG
```
✅ LOG line: `g(S) explains 99.x% of var(log N_ch)` (v6: 99.8 % Pb–Pb, 99.5 % Au–Au), `round trip OK`.
In `targets/trend_fit.png` the orange curve follows the points, and the residuals (lower panels) scatter
around the line without structure. σ_N is ~0.09–0.15, σ_p ~0.04.

### Step 04: smoke test (1 GPU, ~10–15 minutes)

```bash
sbatch slurm/gpu_step.sbatch scripts/04_smoke_test.py $CFG
squeue -u $USER                          # wait until it disappears
tail -3 <run_dir>/LOG.md
```
✅ `OK: zero-init, training and predictions; X s/epoch ... one fold ... takes roughly Y h`.
**Note Y.** It is the time per network for step 05 (v6: 1–1.5 h with ~360 training events per fold; it
grows about linearly with the number of events). If Y is above ~40 h, talk to your supervisor before step 05.
❌ `NO GPU visible`: see [08_troubleshooting.md](08_troubleshooting.md).

### Step 05: train the 5 networks (5 GPUs at once, hours)

**Tell your supervisor before submitting** (5 GPUs for Y hours).

```bash
sbatch slurm/train_cv.sbatch $CFG        # Slurm array 0-4: one fold per GPU
squeue -u $USER                          # 5 tasks: <jobid>_0 ... <jobid>_4
tail -f logs/gouai_cv_<jobid>_0.out      # one line per epoch (Ctrl-C stops watching, not the job)
```
Each line of the `.out` file is one epoch: `loss` (training) and `val_loss` (validation). Expect
`val_loss` to jump up after epoch 1 and to be noisy for 100–200 epochs (the warm-up), then to settle.
Training usually stops between epochs 200 and 350.

✅ When all 5 are done, `LOG.md` has 5 lines `05_train_fold fold k: best epoch ... OOF: N_ch rel. err ...`,
and `cv/fold_k/` contains `best.weights.h5`, `history.csv`, `predictions.csv`, `metrics.json`.
❌ A fold failed or hit the time limit: retrain only that fold, `sbatch --array=3 slurm/train_cv.sbatch $CFG`
(a finished fold is never retrained unless you add `--force`).

### Step 06: cross-validation and Gate 2 (seconds, login node)

```bash
python scripts/06_cv_report.py --config $CFG
```
✅ `cv/report.md` starts with **Gate 2: PASS**. Then look at:
- `learning_curves.png`: every fold's validation loss settles after the warm-up; the kept epoch (●) lies on the settled part;
- `oof_parity.png`: points on the diagonal over three decades of N_ch;
- `oof_residuals.png`: predicted vs true residual along the diagonal (this is the evidence that the network knows more than S);
- the per-class table: compare with Part 1.

❌ **Gate 2: FAIL**: stop. Do not run step 07. See [08_troubleshooting.md](08_troubleshooting.md) "Gate 2 fails".

### Step 07: accuracy on the held-out events (seconds, login node, run ONCE)

```bash
python scripts/07_test_report.py --config $CFG
```
✅ `test/report.md`: headline numbers, per-class tables, and three figures:
- `parity.png` (the paper's Fig. 1 for your sample);
- `error_vs_centrality.png` (network vs trend g(S));
- `fold_spread.png` (disagreement among the 5 networks: the model uncertainty per class).

The network must beat the trend g(S) in every class with enough events.

### Step 08: comparison with v6 on the same events (1 GPU, ~15–30 minutes)

```bash
sbatch slurm/gpu_step.sbatch scripts/08_compare_v6.py $CFG
```
✅ `compare_v6/report.md`:
- **A**: the new held-out events, predicted by the new and by the v6 ensemble. Events v6 trained on are excluded.
- **B**: the v6 held-out events, predicted by both. Its first line must say `v6 predictions reproduce
  the published ones to ~1e-4`.

For each centrality class you get both errors and **new − v6 with a 95 % interval**. Bold means the
difference is unlikely to be chance. These tables are the main result of the project. Interpret them
with [05_evaluation_protocol.md](05_evaluation_protocol.md) §4.

---

## Part 3: the experiments of the project

After Part 2 works, run the experiments of [05_evaluation_protocol.md](05_evaluation_protocol.md) §3.
Each is a new configuration file with its own `run_dir`, repeating steps 01–08:

| experiment | config template | question |
|---|---|---|
| E1 full new sample | `<SYS>_new_entropy.yaml` | how good are networks trained on all new events? (Part 2) |
| E2 same size as v6 | `<SYS>_new_entropy_n450.yaml` | is the gain due to MORE events or to DIFFERENT events? |
| E3 learning curve | copy E2 with `train_subsample:` 900, 1500, ... | how does the error fall with the number of events? |
| E4 (optional) energy input | E1 with `input: energy`, `reference.old_data.x_file: .../x_energy.dat`, `x_kind: energy` | is entropy still as good as energy? |

E2/E3 reuse E1's prepared data and its exact test set (`data.reuse_prepared_from`, `splits.import_from`),
so all experiments are measured on the same held-out events.

Optional steps:
- **09 final network** (`sbatch --time=2-00:00:00 slurm/gpu_step.sbatch scripts/09_train_final.py $CFG`, then re-run 07):
  one network on all training events. v6 found the ensemble better.
- **10 predictions** (`scripts/10_predict.py`): your ensemble on other initial conditions, or the
  1.5-million-event TRENTo production of the paper (only when your supervisor asks for it).

---

## Checklist

| step | command | where | done when |
|---|---|---|---|
| 00 | `scripts/00_check_environment.py` | login | ALL CHECKS PASSED |
| 01 | `scripts/01_prepare_data.py` | login / cpu job | data_report.md read, centrality table noted |
| 02 | `scripts/02_make_splits.py` | login | every class in train and test |
| 03 | `scripts/03_fit_trends.py` | login | > 99 % explained, round trip OK |
| 04 | `slurm/gpu_step.sbatch scripts/04_smoke_test.py` | 1 GPU | OK + time estimate noted |
| 05 | `slurm/train_cv.sbatch` | 5 GPUs | 5 folds in LOG.md |
| 06 | `scripts/06_cv_report.py` | login | Gate 2 PASS, figures looked at |
| 07 | `scripts/07_test_report.py` | login | report read (once) |
| 08 | `slurm/gpu_step.sbatch scripts/08_compare_v6.py` | 1 GPU | v6 reproduced ~1e-4; A and B tables |
