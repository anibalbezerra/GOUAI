# 01 Background: what the networks do and why they are built this way

This page explains the ideas you need to follow the procedure. No previous machine-learning
experience is assumed. Read it once before starting. Return to it whenever a report uses a word you do
not recognize (also see the [glossary](09_glossary.md)).

## 1. The physics question

In a heavy-ion collision, two nuclei (Pb or Au) hit each other at nearly the speed of light. The
overlap region turns into a droplet of quark-gluon plasma, which expands as a nearly perfect fluid,
cools, and finally turns into thousands of hadrons that reach the detector.

Our simulations describe this in stages, the *hybrid model*:

| stage | code | what it computes |
|---|---|---|
| initial condition | TRENTo | where the energy is deposited at the moment of the collision (a 2D map) |
| pre-equilibrium | free streaming | the first ~1 fm/c, before the matter behaves as a fluid |
| fluid | MUSIC | viscous relativistic hydrodynamics with the lattice-QCD equation of state |
| particlization | iS3D | converts the fluid into hadrons at the switching temperature |
| hadronic phase | SMASH | hadron rescattering and decays |

The model parameters are the maximum-a-posteriori values of the JETSCAPE Bayesian analysis. One
hybrid event costs hours of CPU time, so we have only hundreds to a few thousand events per system.

**Centrality.** Experiments sort collisions by their *centrality*: 0–5 % means the 5 % of events with
the most produced particles (head-on collisions), 90–100 % the most grazing ones. To compare with data,
a model must define centrality the same way, **from the final multiplicity N_ch**. That needs millions of
events to fix the class boundaries precisely. Hybrid simulations cannot provide that many. Theorists
therefore usually take centrality from the initial state instead (the total initial entropy S, which
is cheap to compute for millions of TRENTo events). The paper shows that this shortcut puts many events
in the wrong class.

**The solution.** Train a neural network on the few hundred hybrid events to predict N_ch and [pT]
from the initial condition alone. Then generate millions of cheap TRENTo initial conditions and let the
network predict their N_ch and [pT]. The paper did this for 1.5 million events per system in about
2 hours on 8 GPUs, instead of 10⁶–10⁷ CPU-hours.

**Your project.** The new hybrid samples have more events, especially *ultra-central* ones (0–1 %,
0–5 %), where the v6 networks had relatively few training events. You will train the same networks on the
new samples and measure, fairly, how much the extra events improve the predictions (see
[05_evaluation_protocol.md](05_evaluation_protocol.md)).

## 2. The input: an image of the initial state

TRENTo gives the energy density on a 280 × 280 grid with 0.1 fm spacing (28 fm × 28 fm). We convert it
to **entropy density** with the lattice-QCD equation of state (`gouai_train/eos.py`). Entropy is nearly
conserved by the expansion and is closely related to the number of produced particles. The network
receives:

- **image channel 0**, `x / 255`: the absolute density, i.e. how much matter is where;
- **image channel 1**, `x / S`: the same map divided by its sum, i.e. only the *shape* (geometry);
- **scalar input**, `S / max S`: the total entropy (normalized by its largest value among the training events).

N_ch is almost proportional to S. The interesting information, what makes two events with the same S
produce different N_ch or [pT], is in the shape: how compact the fireball is, how lumpy, how elongated.

## 3. The target: only the part S cannot explain

If we plot log N_ch against log S for the training events, they lie on a smooth curve. A cubic
polynomial g(S) fitted to that curve explains more than 99 % of the variance of log N_ch. So we write

```
y = g(S) · exp(d)          d = log y − log g(S)   (the "residual")
```

and the network learns only **d**, standardized as `z = (d − μ)/σ`. Without this step the network would
spend all its effort learning the trivial trend, which spans three orders of magnitude. The small
event-by-event deviations would be lost in the noise. g(S) is fitted on the training events only
(`gouai_train/targets.py`, step 03).

**The key question of every validation step is therefore whether the network beats g(S) alone.**
Two numbers answer it (computed in `gouai_train/metrics.py`):

- **err/σ** = (error of the predicted residual) / (spread of the true residual). 1 means no better than
  g(S), 0.4 means the network removes 1 − 0.4² = 84 % of what S cannot explain.
- **β (calibration slope)**: of the residual the network predicts, how much is real. 1 is ideal,
  0 means pure noise.

## 4. The network

```
image (280×280×2) ─ ResNet-50 ─ attention ─ conv ─ pooling ─ dense layers ─┐
                                                                         ├─ dense ─┬─ head → z_Nch
S / max S ──────── small dense network ─────────────────────────────────┘          └─ head → z_pT
```

- **ResNet-50** is a standard deep convolutional network for images. Here it starts from random
  weights (no pre-training).
- **Spatial attention** lets the network weight the important regions of the map.
- The two **heads** predict z for N_ch and for [pT]. Their last layer starts at zero, so the untrained
  network predicts exactly g(S). Step 04 checks this.
- **Loss**: Huber loss on both z (quadratic for small errors, linear for large ones, robust to outliers).

## 5. Training with few events: the safeguards

**Data augmentation.** N_ch and [pT] do not change if the initial condition is rotated or flipped.
Every epoch therefore shows each training event 13 times: once as is, 8 times flipped and rotated, and
4 times with independent transforms of the two channels (`gouai_train/pipeline.py`). This multiplies
the effective training data and teaches the network the symmetry.

**Held-out test set.** About 25 % of the events are set aside in step 02 and **never** used for training
or for any decision. They are used once, in step 07, to measure the accuracy you will report.

**Cross-validation.** The remaining events are split into 5 *folds*. Network k trains on 4/5 of them and
is monitored on the remaining 1/5 (its *validation* set). Every training event is in exactly one
validation set, so every training event gets one prediction from a network that never saw it. These
are the *out-of-fold (OOF)* predictions, and they are honest estimates of the error on new events.

**Early stopping and warm-up.** The validation loss decides when to stop training (after 80 epochs
without improvement) and which epoch's weights to keep. During the first 100 epochs (warm-up) none of
this happens. The validation loss of a ResNet trained from scratch on few events jumps up after
epoch 1 and needs 60–150 epochs to settle. Without the warm-up, training would stop at a useless early epoch.

**Gate 2.** Before the test set is touched, step 06 checks on the OOF predictions that every fold beats
g(S) alone, that the predicted residuals are real (β), and that training converged. If any criterion
fails, the networks are not used.

**Ensemble.** The final prediction is the geometric mean of the 5 fold networks. Averaging reduces the
noise of the individual networks, and their spread estimates the model uncertainty.

## 6. What "better" means for this project

The error depends strongly on centrality: about 1 % for N_ch in 0–80 %, but 10–30 % in the most peripheral
classes, where N_ch is only a few particles. A test set with many central events will show a smaller
*average* error even if the network is no better. That is why every comparison in this repository is
made **per centrality class**, with centrality from a **minimum-bias** reference, and **on the same
events** for the old and the new networks ([05_evaluation_protocol.md](05_evaluation_protocol.md)).
