# 05 Evaluation protocol: measuring the effect of the new sample fairly

The project asks one question: **do networks trained on the new, larger hybrid sample, which is much
richer in ultra-central events, predict N_ch and [pT] better than the published v6 networks, and where?**
This page fixes how that question is answered *before* any result is seen, so that the answer cannot be
tuned afterwards.

## 1. Rules

1. **The held-out test set is used once per experiment** (step 07) and for the comparison (step 08).
   No setting is chosen by looking at it. If something is changed after step 07 has been run, the
   change and the reason go into the report.
2. **The training procedure is fixed**: `configs/defaults.yaml` = v6. Experiments change only the data
   (which events, how many). Any deviation, e.g. a different number of epochs because training takes too
   long, is a separate, declared experiment.
3. **Gate 2 must pass** before step 07. A failing run is reported as failing, not silently rerun with
   other settings.
4. **Comparisons are made on the same events, per centrality class, with uncertainties.** The rest of
   this page explains why.

## 2. Why a direct comparison of headline numbers is misleading

The paper quotes, for the v6 held-out events, ΔN_ch ≈ 4.8 (Pb–Pb) and 2.8 (Au–Au). Your test report
will show another MAE. Comparing the two numbers says almost nothing:

- **The error depends strongly on centrality.** In minimum-bias classes, the v6 relative error of N_ch
  is ~1–2 % in 0–80 %, ~3 % (Pb–Pb) to ~20 % (Au–Au) in 80–90 %, and ~27 % in 90–100 % (Pb–Pb). The
  *absolute* error (MAE) is largest in central events, where N_ch is ~1000–2000. A test set with more
  central events therefore has a larger MAE and a smaller mean relative error, even for the same network.
- **The test sets differ.** Different events, different sizes, different centrality mix.
- **Few events per class.** The v6 test sets have 3–20 events per class: the class errors themselves
  are uncertain by tens of percent.

Hence:

- **Centrality from a minimum-bias reference** (`reference/centrality/`, built from 1.5 million TRENTo
  events). Percentiles computed inside a hybrid sample are biased. For v6, "20 %" of the sample
  was 12 % of minimum bias, and the sample's "80–100 %" class started at 76 % (Pb–Pb) and 70 % (Au–Au)
  of minimum bias. In
  samples rich in central events the bias is larger still.
- **Class-averaged error**: the per-class mean relative errors averaged over 0–80 %, each class weighted
  by its width. This does not depend on how many test events fall in each class.
- **Paired comparisons**: both ensembles predict the same events (step 08), and the difference of their
  errors is bootstrapped event by event. The event-to-event scatter, common to both models, then cancels.

## 3. The experiments

| id | training events | test events | purpose |
|---|---|---|---|
| **R** | v6 sample, published split | v6 test | reproduction (student guide Part 1, optionally retrained): the environment works |
| **E1** | all new training events | new test (+ v6 test events in the new sample) | the main result |
| **E2** | 450 new training events (as many as v6), stratified on S | **same** as E1 | separates "more events" from "different events" |
| **E3** | 900, 1500, … new training events | **same** as E1 | learning curve: error vs number of events |
| E4 | as E1, energy density as input | same as E1 | optional: input comparison of the paper with the new sample |

E2 and E3 are set up with `splits.import_from: <E1 run>/splits/splits.json` and `splits.train_subsample: N`
(template `configs/examples/<SYS>_new_entropy_n450.yaml`), and `data.reuse_prepared_from: <E1 run>`.

**What each comparison tells you** (all from step 08 set A, or from the step-07 tables of runs that share the test set):

- E1 vs v6, per class: the total effect of the new sample.
- E2 vs v6: the effect of the composition (more central events) at fixed size.
- E1 vs E2: the effect of size at fixed composition.
- E3: the shape of the learning curve. If the error is still falling at the full sample, more hybrid
  events would still help. If it is flat, the limit is elsewhere (network, or fluctuations in the hybrid
  labels themselves).

Comparisons between new runs (E1 vs E2 vs E3) use their test reports, since they share the test set.
For a paired interval between two new runs, use the per-event files `test/predictions.csv` of both runs
with `gouai_train.metrics.paired_bootstrap` (an example is in [06_code_reference.md](06_code_reference.md)).

## 4. Reading the step-08 tables

Set **A** (new held-out events) is the primary comparison: many events, including the ultra-central ones.
Set **B** (the v6 held-out events) asks whether the new networks are at least as good on the events v6
was validated on. It guards against the new networks getting better centrally at the cost of
peripheral accuracy.

For each class:

- `new − v6 < 0` with an interval excluding 0 (bold): the new networks are significantly better there.
- An interval that includes 0: no detectable difference with this many test events. That is *not* the
  same as "equal". Report the interval.
- Watch the ultra-central classes (0–1 %, 1–2 %, 2–5 %): this is where the new sample adds most events.
  In v6 the 0–1 % class had the largest N_ch error below 80 % centrality (Pb–Pb 1.3 %, Au–Au 2.2 %).
- Classes beyond 80 %: v6 had few training events there. Check whether the new sample changed that
  (step 01 table) before interpreting.

Also report from step 07: Gate 2 values, the fold spread per class (model uncertainty), and the
learning curves. A network that is better on average but whose folds disagree more is a warning sign.

## 5. What to report

Fill [results_template.md](results_template.md) for each experiment. Minimum content:

1. Data: number of events per minimum-bias class (step 01), excluded events and why.
2. Split: train/test counts per class (step 02).
3. Gate 2 table (step 06), training times, best epochs.
4. Test: headline numbers, class tables for N_ch and [pT], figures (step 07).
5. Comparison with v6: tables A and B with intervals, figures (step 08).
6. Anything that deviated from this protocol, and why.
