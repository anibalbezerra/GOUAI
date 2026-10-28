# Results: <experiment id, e.g. E1>, <system>, input <entropy/energy>

Student: ……  Date: ……  Repository commit: `git rev-parse --short HEAD` = ……
Config file: ……  run_dir: ……

## 1. Data (step 01: data/data_report.md)

- Usable events: ……  Excluded: …… (reasons: ……)
- Events also in the v6 sample: …… (v6 training ……, v6 held-out ……)

| centrality class | new sample | v6 sample |
|---|---|---|
| (paste the table) | | |

## 2. Split (step 02: splits/splits_report.md)

Train …… / test ……; v6 held-out events forced into test: ……; subsample: …… (E2/E3 only)

## 3. Cross-validation (step 06: cv/report.md)

Gate 2: PASS / FAIL. Pooled OOF β_N …… err/σ_N …… β_p …… err/σ_p ……
Best epochs: ……  Training time per fold: …… h  Jobs: ……
Anything unusual in the learning curves: ……

## 4. Held-out test (step 07: test/report.md)

| | N_ch MAE | N_ch rel. err | N_ch class-avg 0–80 % | [pT] MAE (MeV) | [pT] class-avg 0–80 % |
|---|---|---|---|---|---|
| ensemble | | | | | |
| trend g(S) | | | | | |

Per-class tables: (paste or attach `test/report.md`). Figures: `parity`, `error_vs_centrality`, `fold_spread`.

## 5. Comparison with v6 (step 08: compare_v6/report.md)

| set | n | N_ch class-avg: new | v6 | new − v6 [95 % CI] | [pT] class-avg: new | v6 | new − v6 [95 % CI] |
|---|---|---|---|---|---|---|---|
| A (new test) | | | | | | | |
| B (v6 test) | | | | | | | |

Ultra-central classes (0–1 %, 1–2 %, 2–5 %), N_ch: new …… / v6 …… / difference ……
v6 reproduction on set B: ……

## 6. Deviations from the protocol

(none / describe what was changed, when, and why)

## 7. Interpretation (3–5 sentences)
