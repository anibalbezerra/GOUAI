#!/usr/bin/env python3
"""
STEP 06 - cross-validation report and Gate 2                          (CPU, seconds)

    python scripts/06_cv_report.py --config configs/my_experiment.yaml

Collects the out-of-fold (OOF) predictions of the 5 networks: every outer-train event predicted by
the one network that did not train on it. On these honest predictions Gate 2 asks, with the
thresholds of the gates section of the config (identical to v6):

  (a) N_ch residual (events with N_ch >= 10):
        pooled calibration slope beta_N >= 0.3 and pooled err/sigma_N < 1,
        and in EVERY fold r_N > 0 and err/sigma_N < 1          -> each network beats g(S) alone
  (b) [pT] residual: pooled beta_p >= 0.3, pooled err/sigma_p < 1, r_p > 0 in every fold
  (c) convergence: in every fold the best validation loss <= 0.5 x its epoch-1 value, and early
      stopping fired before the epoch limit (the optimum was reached, not cut off)

If Gate 2 fails, do NOT go on to the test set: read docs/08_troubleshooting.md ("Gate 2 fails").

Writes <run_dir>/cv/report.md, cv/oof_metrics.json, cv/final_plan.json (epochs for step 09) and
figures cv/learning_curves, cv/oof_parity, cv/oof_residuals.
"""
import os
import sys

import numpy as np
import pandas as pd

import _path  # noqa: F401
from gouai_train.config import Run
from gouai_train.targets import ResidualSpec
from gouai_train.centrality import CentralityTable
from gouai_train import metrics as MT
from gouai_train import plots
from gouai_train.style import NEW

STEP = '06_cv_report'


def main():
    run = Run.from_args(__doc__)
    cfg = run.cfg
    split = run.read_json('splits/splits.json')
    nf = cfg['splits']['n_folds']
    for k in range(nf):
        run.require(f'cv/fold_{k}/metrics.json', f'05_train_fold.py --fold {k}')
    gref = ResidualSpec.from_dict(run.read_json('targets/gref.json'))
    cent = CentralityTable(cfg['evaluation']['centrality_table'])

    oof, fold_m, fold_info, hists = [], [], [], {}
    for k in range(nf):
        d = os.path.join(run.dir, 'cv', f'fold_{k}')
        m = run.read_json(f'cv/fold_{k}/metrics.json')
        p = pd.read_csv(os.path.join(d, 'predictions.csv'))
        oof.append(p[p.part == 'oof'])
        fold_m.append(m['oof'])
        fold_info.append(m['info'])
        hists[f'fold {k}'] = pd.read_csv(os.path.join(d, 'history.csv'))
    oof = pd.concat(oof)
    assert sorted(oof.idx) == sorted(split['outer_train']), 'OOF predictions do not cover the training events exactly once'
    yt = {'nch': oof.nch_true.values, 'pt': oof.pt_true.values}
    yp = {'nch': oof.nch_pred.values, 'pt': oof.pt_pred.values}
    m_oof = MT.evaluate(oof.S.values, yt, yp, gref, cent(oof.S.values), cfg)

    g, fl, emax = cfg['gates'], cfg['targets']['nch_floor'], cfg['training']['epochs']
    oN, oP = m_oof['nch']['nch_ge_floor'], m_oof['pt']['all']
    fN = [fm['nch']['nch_ge_floor'] for fm in fold_m]
    fP = [fm['pt']['all'] for fm in fold_m]
    conv = [i['best_val_loss'] / i['val_loss_epoch1'] for i in fold_info]
    fmt = lambda xs: ', '.join(f'{x:.3f}' for x in xs)
    crit = {
        f'(a) pooled β_N (N_ch≥{fl:g}) ≥ {g["gate2_beta_nch_min"]}': (oN['beta_cal'] >= g['gate2_beta_nch_min'], f'{oN["beta_cal"]:.3f}'),
        f'(a) pooled err/σ_N (N_ch≥{fl:g}) < {g["gate2_err_ratio_nch_max"]}': (oN['err_ratio'] < g['gate2_err_ratio_nch_max'], f'{oN["err_ratio"]:.3f}'),
        '(a) r_N > 0 in every fold': (all(x['r'] > 0 for x in fN), fmt(x['r'] for x in fN)),
        f'(a) err/σ_N < {g["gate2_err_ratio_nch_max"]} in every fold': (all(x['err_ratio'] < g['gate2_err_ratio_nch_max'] for x in fN), fmt(x['err_ratio'] for x in fN)),
        f'(b) pooled β_p ≥ {g["gate2_beta_pt_min"]}': (oP['beta_cal'] >= g['gate2_beta_pt_min'], f'{oP["beta_cal"]:.3f}'),
        f'(b) pooled err/σ_p < {g["gate2_err_ratio_pt_max"]}': (oP['err_ratio'] < g['gate2_err_ratio_pt_max'], f'{oP["err_ratio"]:.3f}'),
        '(b) r_p > 0 in every fold': (all(x['r'] > 0 for x in fP), fmt(x['r'] for x in fP)),
        f'(c) best / epoch-1 val loss ≤ {g["gate2_convergence_ratio"]} in every fold': (all(c <= g['gate2_convergence_ratio'] for c in conv), fmt(conv)),
        f'(c) early stopping before {emax} epochs in every fold': (all(i['epochs_run'] < emax for i in fold_info), ', '.join(str(i['epochs_run']) for i in fold_info)),
    }
    passed = all(v[0] for v in crit.values())
    best_epochs = [i['best_epoch'] for i in fold_info]
    run.write_json('cv/final_plan.json', {'epochs': int(np.median(best_epochs)), 'fold_best_epochs': best_epochs})
    run.write_json('cv/oof_metrics.json', {'oof': m_oof, 'gate2': crit, 'passed': passed})

    rows = [MT.flat_row('OOF, all folds pooled', m_oof)] + [MT.flat_row(f'fold {k} (validation)', fm) for k, fm in enumerate(fold_m)]
    rep = [f'# Cross-validation: {cfg["label"]}, input {cfg["input"]}\n',
           f'**Gate 2: {"PASS" if passed else "FAIL"}**\n',
           '| criterion | pass | value |', '|---|---|---|',
           *[f'| {k} | {"✔" if v[0] else "✘"} | {v[1]} |' for k, v in crit.items()],
           '\nOOF = every training event predicted by the network that did not train on it. '
           'β = calibration slope of the residual beyond g(S) (1 = real), err/σ < 1 = better than g(S) alone.\n',
           '## All events\n', MT.markdown_table(rows), '',
           f'## N_ch ≥ {fl:g}\n', MT.markdown_table([MT.flat_row('OOF pooled', m_oof, 'nch_ge_floor')]), '',
           '## Mean relative error per minimum-bias centrality class (OOF)\n',
           '### N_ch\n', MT.class_table({'OOF': m_oof}, 'nch'), '',
           '### [pT]\n', MT.class_table({'OOF': m_oof}, 'pt'), '',
           f'Best epochs {best_epochs} (warm-up {cfg["training"]["warmup_epochs"]}); '
           f'training time per fold {fmt(i["wall_s"] / 3600 for i in fold_info)} h.\n',
           '![learning curves](learning_curves.png)\n', '![OOF parity](oof_parity.png)\n',
           '![OOF residuals](oof_residuals.png)\n']
    with open(run.path('cv', 'report.md'), 'w') as f:
        f.write('\n'.join(rep))

    plots.learning_curves(hists, cfg['training']['warmup_epochs'], run.path('cv', 'learning_curves'),
                          {f'fold {k}': e for k, e in enumerate(best_epochs)}, f'{cfg["label"]}: learning curves')
    plots.parity({'OOF': (yt, yp, NEW, 'o')}, run.path('cv', 'oof_parity'), f'{cfg["label"]}: out-of-fold')
    lg = {n: gref.trends[n].log_g(oof.S.values) for n in ('nch', 'pt')}
    plots.residual_recovery({'OOF': ({n: np.log(yt[n]) - lg[n] for n in lg}, {n: np.log(yp[n]) - lg[n] for n in lg}, NEW)},
                            run.path('cv', 'oof_residuals'), f'{cfg["label"]}: residual beyond g(S)')

    run.log(STEP, f'Gate 2 {"PASS" if passed else "FAIL"}: OOF β_N={oN["beta_cal"]:.3f} err/σ_N={oN["err_ratio"]:.3f} '
                  f'β_p={oP["beta_cal"]:.3f} err/σ_p={oP["err_ratio"]:.3f}; N_ch class-avg rel. err '
                  f'{100 * m_oof["nch"]["class_avg"]["mean_rel_err"]:.2f}% (0-80%). Read cv/report.md')
    if not passed:
        print('\nGate 2 FAILED: do not use these networks. See docs/08_troubleshooting.md.\n')
        sys.exit(1)


if __name__ == '__main__':
    main()
