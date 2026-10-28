# GOUAI v2 — Pb-Pb, sqrt(s_NN) = 2.76 TeV, entropy density input

**Predicts:** the event-by-event charged-particle multiplicity N_ch and mean transverse momentum [pT] (GeV) of
charged particles with pT >= 0.15 GeV at midrapidity, as given by the JETSCAPE hybrid model (TRENTo + free
streaming + MUSIC + iS3D + SMASH, maximum-a-posteriori parameters), from the TRENTo initial condition.

**Model:** 5-fold ensemble (fold_0, fold_1, fold_2, fold_3, fold_4), combined by geometric mean.
Each network predicts the standardized residual beyond a fixed trend g(S) in the total input S (see
`gouai/targets.py`, `gouai/network.py`).

**Inputs** (must match exactly):
1. TRENTo energy density on a 280x280 grid, `--grid-max 14 --grid-step 0.1`, with `-p 0.063 -k 1.05 -x 6.28 -w 1.12 -d 1.437425 -n 14.2 --grid-max 14.0 --grid-step 0.1 Pb Pb`.
2. entropy density from the TRENTo energy density: cubic interpolation of the HRG/HotQCD table, s = 0 below its minimum (gouai.eos).
3. ch0 = x/255, ch1 = x/S (S = pixel sum of the 280x280 grid); scalar input S/max_S, max_S = 1239434.101.

**Training data:** 450 hybrid events spanning 0-100% centrality (centrality-stratified;
few events beyond 90%), S in [2367.5, 1.23943e+06]. Predictions outside this
range rely on the linearly extrapolated trend. Validation: 169 held-out events never used for
training or model selection.

## Held-out accuracy (ensemble)
| | MAE | mean relative error | relative error (N_ch >= 10) |
|---|---|---|---|
| N_ch | 4.76 | 2.95% | 1.14% |
| [pT] | 5.53 MeV | 0.91% | |

Residual beyond g(S), N_ch >= 10: r = 0.94, calibration slope beta = 1.07,
rms error / residual spread = 0.34.

Per centrality class (from the initial entropy):

| class | events | N_ch | [pT] |
|---|---|---|---|
| 0-10% | 20 | 1.1% | 0.6% |
| 10-20% | 10 | 0.5% | 0.5% |
| 20-30% | 23 | 1.1% | 0.9% |
| 30-40% | 15 | 1.1% | 0.4% |
| 40-50% | 15 | 1.2% | 0.5% |
| 50-60% | 20 | 1.1% | 0.8% |
| 60-70% | 16 | 1.2% | 1.3% |
| 70-80% | 15 | 1.2% | 0.7% |
| 80-100% | 35 | 10.0% | 1.6% |

The most peripheral class (80-100%, N_ch < ~10) is the least accurate and least represented in training.

## Files
`ensemble.json`, `fold_k/spec.json` (trend + standardization of fold k), `fold_k/model.weights.h5`
(downloaded, see `scripts/download_models.py`), `fold_k/history.csv`, `fold_k/metrics.json`, `metrics.json`,
`heldout_predictions.csv`, `train_oof_predictions.csv`.
