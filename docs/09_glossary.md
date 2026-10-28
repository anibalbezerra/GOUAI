# 09 Glossary

**Augmentation**: making extra training examples by flipping and rotating the initial condition, which
does not change N_ch or [pT]. 13 copies of every event per epoch (`pipeline.py`).

**Batch**: the 256 examples used for one update of the weights.

**β (beta_cal), calibration slope**: cov(d_true, d_pred)/var(d_pred). Of the residual the network
predicts, the fraction that is real. 1 = calibrated, 0 = noise, > 1 = the network is too timid.

**Centrality**: the percentage of minimum-bias events that are more central (have larger S here). 0 % =
head-on. In this repository it always comes from the minimum-bias table (`reference/centrality/`), never
from percentiles inside the hybrid sample.

**Class-averaged error**: the per-class mean relative errors averaged over 0–80 %, weighted by class
width. Independent of how many test events fall in each class.

**Cross-validation (CV), fold**: the training events are split into 5 parts. Network k trains on 4 of
them and is validated on part k.

**Early stopping**: training stops when the validation loss has not improved for `es_patience` (80) epochs.
The weights of the best epoch are kept.

**Ensemble**: the 5 fold networks used together. The prediction is the geometric mean of their predictions.

**Epoch**: one pass over all (augmented) training examples.

**err/σ (err_ratio)**: rmse of the predicted residual / spread of the true residual. < 1 means better
than g(S) alone.

**Event table**: `data/event_table.csv`, one row per usable event; `idx` is the event number used everywhere.

**g(S), trend**: the smooth dependence of log N_ch (log [pT]) on log S: a cubic polynomial fitted to the
training events (`targets.py`). **g_ref** is the one fitted to all training events, used as the common
reference for residual metrics.

**Gate 2**: the cross-validation acceptance test (step 06). It must pass before the test set is used.

**Held-out test set, outer test**: events never used for training or for any decision. Used once, in
step 07 (and in step 08).

**Huber loss**: quadratic for small errors, linear for large ones (δ = 1).

**Hybrid model**: TRENTo + free streaming + MUSIC + iS3D + SMASH, the full simulation the networks emulate.

**Learning rate**: the size of the weight updates (3 × 10⁻⁴, halved on plateaus).

**MAE**: mean absolute error, in N_ch units or MeV for [pT].

**Mean relative error**: mean of |prediction/truth − 1|.

**Minimum bias**: all collisions, without selection, in the proportion nature produces them.

**N_ch**: number of charged particles with pT ≥ 0.15 GeV at midrapidity (|η| < 0.5).

**OOF, out-of-fold prediction**: the prediction for a training event by the fold network that did not
train on it. An honest error estimate without touching the test set.

**[pT]**: mean transverse momentum of those particles, in GeV.

**Residual d, z**: d = log y − log g(S), the part of the observable S does not explain; z = (d − μ)/σ is
what the network outputs.

**Run directory (run_dir)**: the folder with all outputs of one experiment.

**S**: total initial entropy (pixel sum of the entropy-density grid), or total energy for energy input.

**Spec (spec.json)**: the trend g(S) and the normalization (μ, σ) belonging to one trained network.

**Split**: which events are training/test, and the folds (`splits/splits.json`).

**TRENTo**: the initial-condition model. It is cheap, so millions of events can be generated.

**v6**: the published GOUAI networks (paper and GOUAI repository). Their training data, split and
predictions are in `reference/v6/`.

**Validation set**: the events used to stop training and to choose the epoch. Unlike the test set, they
influence the network indirectly.

**Warm-up**: the first 100 epochs, during which no early stopping, learning-rate cut or checkpoint happens.
