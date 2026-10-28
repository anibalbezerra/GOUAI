# Reference data

Small files needed to reproduce and to compare with the published GOUAI v6 networks. Nothing here is
produced by the step scripts. Do not edit these files.

## `centrality/<system>_<input>.csv`

Minimum-bias centrality tables: for every 0.01 % of centrality, the smallest total initial entropy
(`*_entropy.csv`) or energy (`*_energy.csv`) of an event in that most-central fraction. They were built with
`tools/make_centrality_table.py` from the 1.5 × 10⁶ TRENTo events per system of the v6 production
(the TRENTo parameters of the hybrid events: `configs/examples/*`, `production.trento_args`). They are valid
only for initial conditions generated with those parameters, grid and equation of state.

## `v6/`

The published v6 networks, in the layout of the GOUAI repository's `trained_models/`, **without weights**:

```
v6/weights_manifest.json                      file names and SHA-256 of the 20 weight files (Zenodo)
v6/<system>/splits.json                       the held-out events (outer_test) and the 5 folds
v6/<system>/<input>_pt/ensemble.json          training settings, max S used to normalize the scalar input
v6/<system>/<input>_pt/MODEL_CARD.md          what the networks predict, inputs, accuracy
v6/<system>/<input>_pt/metrics.json           published held-out metrics
v6/<system>/<input>_pt/heldout_predictions.csv     ensemble prediction for every held-out event
v6/<system>/<input>_pt/train_oof_predictions.csv   out-of-fold prediction for every training event
v6/<system>/<input>_pt/fold_<k>/spec.json     trend g(S) and normalization of network k
v6/<system>/<input>_pt/fold_<k>/history.csv   its learning curve
v6/<system>/<input>_pt/fold_<k>/metrics.json  its training info
v6/<system>/<input>_pt/fold_<k>/model.weights.h5   <- placed by scripts/fetch_v6_weights.py (not in git)
```

Event indices (`idx_raw`, splits) are line numbers of the original `x_*.dat` / `Y_eta0p5_pt.dat` files.
The `centrality_from_S_percent` column of the prediction files is the percentile inside the hybrid sample,
as used in the v6 reports. This repository uses minimum-bias centrality instead
(docs/05_evaluation_protocol.md §2).
