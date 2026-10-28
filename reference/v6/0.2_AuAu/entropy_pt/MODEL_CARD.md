# GOUAI v2 — Au-Au, sqrt(s_NN) = 200 GeV, entropy density input

**Predicts:** the event-by-event charged-particle multiplicity N_ch and mean transverse momentum [pT] (GeV) of
charged particles with pT >= 0.15 GeV at midrapidity, as given by the JETSCAPE hybrid model (TRENTo + free
streaming + MUSIC + iS3D + SMASH, maximum-a-posteriori parameters), from the TRENTo initial condition.

**Model:** 5-fold ensemble (fold_0, fold_1, fold_2, fold_3, fold_4), combined by geometric mean.
Each network predicts the standardized residual beyond a fixed trend g(S) in the total input S (see
`gouai/targets.py`, `gouai/network.py`).

**Inputs** (must match exactly):
1. TRENTo energy density on a 280x280 grid, `--grid-max 14 --grid-step 0.1`, with `-p 0.063 -k 1.05 -x 6.28 -w 1.12 -d 1.437425 -n 5.73 --grid-max 14.0 --grid-step 0.1 Au Au`.
2. entropy density from the TRENTo energy density: cubic interpolation of the HRG/HotQCD table, s = 0 below its minimum (gouai.eos).
3. ch0 = x/255, ch1 = x/S (S = pixel sum of the 280x280 grid); scalar input S/max_S, max_S = 559397.1282.

**Training data:** 449 hybrid events spanning 0-100% centrality (centrality-stratified;
few events beyond 90%), S in [2295.3, 559397]. Predictions outside this
range rely on the linearly extrapolated trend. Validation: 135 held-out events never used for
training or model selection.

## Held-out accuracy (ensemble)
| | MAE | mean relative error | relative error (N_ch >= 10) |
|---|---|---|---|
| N_ch | 2.82 | 2.89% | 1.39% |
| [pT] | 5.08 MeV | 0.89% | |

Residual beyond g(S), N_ch >= 10: r = 0.93, calibration slope beta = 1.13,
rms error / residual spread = 0.38.

Per centrality class (from the initial entropy):

| class | events | N_ch | [pT] |
|---|---|---|---|
| 0-10% | 18 | 1.7% | 0.6% |
| 10-20% | 9 | 1.3% | 0.7% |
| 20-30% | 17 | 0.9% | 0.7% |
| 30-40% | 12 | 1.5% | 0.9% |
| 40-50% | 10 | 0.9% | 0.7% |
| 50-60% | 12 | 1.0% | 0.6% |
| 60-70% | 13 | 1.7% | 0.9% |
| 70-80% | 15 | 1.8% | 0.9% |
| 80-100% | 29 | 8.4% | 1.4% |

The most peripheral class (80-100%, N_ch < ~10) is the least accurate and least represented in training.

## Files
`ensemble.json`, `fold_k/spec.json` (trend + standardization of fold k), `fold_k/model.weights.h5`
(downloaded, see `scripts/download_models.py`), `fold_k/history.csv`, `fold_k/metrics.json`, `metrics.json`,
`heldout_predictions.csv`, `train_oof_predictions.csv`.
