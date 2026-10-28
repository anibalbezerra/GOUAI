#!/usr/bin/env python3
"""
STEP 07 - accuracy on the held-out test set                           (CPU, seconds)

    python scripts/07_test_report.py --config configs/my_experiment.yaml

Run this ONCE, after Gate 2 has passed. The test events were never used for training, early stopping
or any choice, so these are the numbers to report. Looking at them and then changing the training
would turn the test set into a validation set: if you change anything after this step, say so.

Models compared on the same test events:
    ensemble     geometric mean of the 5 fold networks (the published GOUAI prediction)
    final        the single network of step 09, if it was trained
    trend g(S)   the trend alone: what S knows without any network (the baseline to beat)

Writes <run_dir>/test/report.md, test/metrics.json, test/predictions.csv (one row per test event, with
the spread of the 5 networks) and the figures test/parity, test/error_vs_centrality, test/fold_spread.
"""
import os

import numpy as np
import pandas as pd

import _path  # noqa: F401
from gouai_train.config import Run, SYSTEMS
from gouai_train.targets import ResidualSpec
from gouai_train.centrality import CentralityTable
from gouai_train import metrics as MT
from gouai_train import plots
from gouai_train.style import NEW, V6, TREND, SYSTEM_STYLE

STEP = '07_test_report'


def main():
    run = Run.from_args(__doc__)
    cfg = run.cfg
    run.require('cv/oof_metrics.json', '06_cv_report.py')
    if not run.read_json('cv/oof_metrics.json')['passed']:
        raise SystemExit('Gate 2 did not pass (cv/report.md): fix the training before looking at the test set.')
    split = run.read_json('splits/splits.json')
    te = np.array(split['outer_test'])
    tab = pd.read_csv(run.path('data', 'event_table.csv')).set_index('idx')
    S = tab[f'S_{cfg["input"]}'].loc[te].values
    yt = {'nch': tab.nch.loc[te].values, 'pt': tab.pt.loc[te].values}
    gref = ResidualSpec.from_dict(run.read_json('targets/gref.json'))
    cent_t = CentralityTable(cfg['evaluation']['centrality_table'])
    cent = cent_t(S)

    tests = []
    for k in range(cfg['splits']['n_folds']):
        p = pd.read_csv(os.path.join(run.dir, 'cv', f'fold_{k}', 'predictions.csv'))
        tests.append(p[p.part == 'test'].set_index('idx').loc[te])
    members = np.stack([np.stack([np.log(t.nch_pred.values), np.log(t.pt_pred.values)], 1) for t in tests])
    preds = {'ensemble': {'nch': np.exp(members[:, :, 0].mean(0)), 'pt': np.exp(members[:, :, 1].mean(0))}}
    fin = os.path.join(run.dir, 'final', 'predictions.csv')
    if os.path.exists(fin):
        fp = pd.read_csv(fin).set_index('idx').loc[te]
        preds['final'] = {'nch': fp.nch_pred.values, 'pt': fp.pt_pred.values}
    preds['trend g(S)'] = {n: np.exp(gref.trends[n].log_g(S) + gref.mu[n]) for n in ('nch', 'pt')}
    mets = {k: MT.evaluate(S, yt, p, gref, cent, cfg) for k, p in preds.items()}
    run.write_json('test/metrics.json', mets)

    spread = members.std(axis=0, ddof=1)
    pd.DataFrame({'idx': te, 'event_id': tab.event_id.loc[te].values, 'centrality': cent, 'S': S,
                  'nch_true': yt['nch'], 'nch_pred': preds['ensemble']['nch'], 'nch_fold_spread_rel': spread[:, 0],
                  'pt_true_GeV': yt['pt'], 'pt_pred_GeV': preds['ensemble']['pt'], 'pt_fold_spread_rel': spread[:, 1]}
                 ).to_csv(run.path('test', 'predictions.csv'), index=False, float_format='%.7g')

    fl = cfg['targets']['nch_floor']
    e = mets['ensemble']
    rep = [f'# Held-out test: {cfg["label"]}, input {cfg["input"]}\n',
           f'{len(te)} test events, never used for training or model selection. Centrality = minimum-bias '
           f'centrality from S (`{os.path.basename(cfg["evaluation"]["centrality_table"])}`).\n',
           '## Headline numbers (ensemble)\n',
           f'- N_ch: MAE **{e["nch"]["all"]["mae"]:.2f}**, mean relative error {100 * e["nch"]["all"]["mean_rel_err"]:.2f}% '
           f'(N_ch ≥ {fl:g}: {100 * e["nch"]["nch_ge_floor"]["mean_rel_err"]:.2f}%), '
           f'class-averaged {e["nch"]["class_avg"]["range"]}: **{100 * e["nch"]["class_avg"]["mean_rel_err"]:.2f}%**',
           f'- [pT]: MAE **{e["pt"]["all"]["mae"]:.2f} MeV**, mean relative error {100 * e["pt"]["all"]["mean_rel_err"]:.2f}%, '
           f'class-averaged {e["pt"]["class_avg"]["range"]}: **{100 * e["pt"]["class_avg"]["mean_rel_err"]:.2f}%**',
           f'- residual beyond g(S): β_N = {e["nch"]["nch_ge_floor"]["beta_cal"]:.2f}, err/σ_N = {e["nch"]["nch_ge_floor"]["err_ratio"]:.2f}; '
           f'β_p = {e["pt"]["all"]["beta_cal"]:.2f}, err/σ_p = {e["pt"]["all"]["err_ratio"]:.2f}\n',
           'The plain mean over test events depends on how many test events each centrality class has; '
           'the class-averaged number does not, and is the one to compare between samples.\n',
           '## All test events\n', MT.markdown_table([MT.flat_row(k, v) for k, v in mets.items()]), '',
           f'## N_ch ≥ {fl:g}\n', MT.markdown_table([MT.flat_row(k, v, 'nch_ge_floor') for k, v in mets.items()]), '',
           '## Mean relative error per centrality class\n', '### N_ch\n', MT.class_table(mets, 'nch'), '',
           '### [pT]\n', MT.class_table(mets, 'pt'), '',
           '![parity](parity.png)\n', '![error vs centrality](error_vs_centrality.png)\n',
           '![fold spread](fold_spread.png)\n']
    with open(run.path('test', 'report.md'), 'w') as f:
        f.write('\n'.join(rep))

    st = SYSTEM_STYLE[cfg['system']]
    plots.parity({SYSTEMS[cfg['system']]['tex']: (yt, preds['ensemble'], st['color'], st['marker'])},
                 run.path('test', 'parity'))
    colors = {'ensemble': NEW, 'final': V6, 'trend g(S)': TREND}
    plots.error_vs_centrality({k: (m, colors[k]) for k, m in mets.items()}, run.path('test', 'error_vs_centrality'),
                              f'{cfg["label"]}: held-out test')
    plots.fold_spread(cent, members, cfg['evaluation']['centrality_edges'], run.path('test', 'fold_spread'),
                      f'{cfg["label"]}: disagreement of the 5 networks')
    run.log(STEP, f'ensemble on {len(te)} test events: N_ch MAE {e["nch"]["all"]["mae"]:.2f}, class-avg rel. err '
                  f'{100 * e["nch"]["class_avg"]["mean_rel_err"]:.2f}%; [pT] MAE {e["pt"]["all"]["mae"]:.2f} MeV, '
                  f'class-avg {100 * e["pt"]["class_avg"]["mean_rel_err"]:.2f}%. Read test/report.md')


if __name__ == '__main__':
    main()
