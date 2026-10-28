# GOUAI-train

Train, validate and compare the **GOUAI** neural networks on new hybrid-model events.

GOUAI (*Global Observables Using Artificial Intelligence*) predicts, event by event, the charged-particle
multiplicity N_ch and the mean transverse momentum [pT] of a heavy-ion collision from its initial
condition (the TRENTo entropy or energy density). The networks replace the full hybrid simulation
(free streaming → MUSIC viscous hydrodynamics → iS3D particlization → SMASH hadronic transport)
for these two observables. That makes millions of events, and therefore centrality defined from
the final multiplicity as in experiment, affordable.

This repository is the training side of GOUAI. It reproduces, step by step, the procedure that
produced the published **v6** networks of the paper, so that the same procedure can be applied to new
hybrid samples and the results compared with v6 in a fair, controlled way:

> A. Bezerra and F. G. Gardim, *Final-state centrality in relativistic heavy-ion simulations from neural networks*,
> [arXiv:2610.13828](https://arxiv.org/abs/2610.13828) [nucl-th] (2026).
> Trained networks and inference code: [GOUAI](https://github.com/anibalbezerra/GOUAI).

| | |
|---|---|
| Systems | Pb–Pb at √s_NN = 2.76 TeV, Au–Au at √s_NN = 200 GeV (JETSCAPE maximum-a-posteriori parameters) |
| Input | 280 × 280 grid of the initial entropy density (or energy density), plus its total S |
| Output | N_ch and [pT] of charged particles with pT ≥ 0.15 GeV, \|η\| < 0.5 |
| Network | ResNet-50 + spatial attention + scalar branch, two residual heads (`gouai_train/model.py`) |
| Targets | residual beyond a smooth trend g(S), standardized (`gouai_train/targets.py`) |
| Validation | held-out test set + 5-fold cross-validation; prediction = geometric mean of the 5 networks |
| v6 accuracy | N_ch ≈ 1.1–1.4 %, [pT] ≈ 0.8 % (mean relative error, 0–80 % centrality, held-out events) |

## Quick start

Read [docs/04_student_guide.md](docs/04_student_guide.md) before running anything. In short:

```bash
cd <a disk the compute nodes can see>                     # see docs/02_installation.md §1
git clone https://github.com/anibalbezerra/GOUAI-train.git && cd GOUAI-train
source <your python environment>/bin/activate             # docs/02_installation.md §2
cp configs/examples/PbPb_new_entropy.yaml configs/my_PbPb.yaml    # edit the lines marked CHANGE
CFG=configs/my_PbPb.yaml

python scripts/00_check_environment.py --config $CFG       # all checks must pass
python scripts/01_prepare_data.py      --config $CFG       # read and check the events
python scripts/02_make_splits.py       --config $CFG       # held-out test set + 5 folds
python scripts/03_fit_trends.py        --config $CFG       # trend g(S) and normalizations
sbatch slurm/gpu_step.sbatch scripts/04_smoke_test.py $CFG # 10-minute GPU test
sbatch slurm/train_cv.sbatch $CFG                          # 5 networks, 5 GPUs, hours
python scripts/06_cv_report.py         --config $CFG       # cross-validation + Gate 2
python scripts/07_test_report.py       --config $CFG       # held-out accuracy (run once)
python scripts/fetch_v6_weights.py --from-dir <folder with the v6 weight files>
sbatch slurm/gpu_step.sbatch scripts/08_compare_v6.py $CFG # new vs published v6, same events
```

Every step writes its results to the experiment's `run_dir` and one line to `<run_dir>/LOG.md`.
Nothing runs automatically. You start each step yourself and check its output before the next one.

## Documentation

| | |
|---|---|
| [01 Background](docs/01_background.md) | the physics question, and the machine-learning ideas used (for newcomers) |
| [02 Installation](docs/02_installation.md) | environment, cluster rules, GPU check |
| [03 Data format](docs/03_data_format.md) | what the hybrid events must look like; checks done on them |
| [04 Student guide](docs/04_student_guide.md) | **the step-by-step procedure**, with what to check after every step |
| [05 Evaluation protocol](docs/05_evaluation_protocol.md) | how to compare with v6 fairly; the experiments of the project |
| [06 Code reference](docs/06_code_reference.md) | every module and script, what each important part does and why |
| [07 Adding inputs](docs/07_adding_inputs.md) | giving the network new inputs (channels, scalars) or new targets |
| [08 Troubleshooting](docs/08_troubleshooting.md) | errors, failed gates, slow jobs |
| [09 Glossary](docs/09_glossary.md) | terms used in the code and the reports |

## Repository map

```
configs/defaults.yaml         every setting, with the published v6 values (do not edit; override per experiment)
configs/examples/             experiment templates: new data (PbPb/AuAu), v6 reproduction, learning-curve runs
gouai_train/                  the library (documented module by module in docs/06_code_reference.md)
scripts/00 ... 10             the steps, in order; each prints what it did and what to read
scripts/fetch_v6_weights.py   places the published v6 weights for step 08 (not in git: 20 x 135 MB)
slurm/                        job scripts for the cluster (CPU step, GPU step, 5-fold array, TRENTo array)
reference/centrality/         minimum-bias centrality tables (1.5 M TRENTo events per system and input)
reference/v6/                 the published v6 networks without weights: splits, per-fold trends, metrics,
                              learning curves, predictions for every training and held-out event
tools/                        maintenance: build centrality tables, check that v6 is reproduced exactly
tests/                        python -m pytest tests -q   (1 minute, CPU)
```

## Relation to the GOUAI repository

The [GOUAI repository](https://github.com/anibalbezerra/GOUAI) distributes the **trained** networks and a small inference package
(`GouaiModel('PbPb').predict_trento(grids)`). This repository **trains** networks on new hybrid
events. Its networks are built, trained and stored the same way as GOUAI's, so a network trained here
can be published through GOUAI in the same format.

## Reproducibility statement

With the original events and the published split (`configs/examples/reproduce_v6_*.yaml`), this code
reproduces the published v6 trend fits to 2 × 10⁻¹⁰ and feeds the published networks exactly as in
training. On the same device its predictions agree with the original code to 3 × 10⁻⁷, and the
held-out metrics of the paper are recovered exactly (`tools/check_v6_reproduction.py`).
Training on GPUs is not bit-reproducible (cuDNN). A retrained ensemble matches the published one
statistically, not weight by weight.

## Citation and license

If you use this code, cite the paper above and the GOUAI repository (`CITATION.cff`). MIT license.
