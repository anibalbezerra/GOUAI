#!/usr/bin/env python3
"""
STEP 03 - fit the trend g(S) and the normalizations                        (CPU, seconds)

    python scripts/03_fit_trends.py --config configs/my_experiment.yaml

For each fold k, the trend g(S) and the standardization (mu, sigma) of the residual are fitted on the
fold's TRAINING events only (never on its validation events, never on the test set):

    <run_dir>/targets/fold_<k>_spec.json    used by network k (step 05)
    <run_dir>/targets/final_spec.json       fitted on all outer-train events (final model, step 09)
    <run_dir>/targets/gref.json             the same fit, used as the common reference trend g_ref
                                            for the residual metrics of every model
    <run_dir>/targets/meta.json             channels, scalar inputs and their normalization
                                            (max over the outer-train events), S range of training
    <run_dir>/targets/trend_fit.png         g(S) and the residuals of the training events

Also checks that the target transform is exactly invertible for every event (round trip y -> z -> y).
"""
import numpy as np
import pandas as pd

import _path  # noqa: F401
from gouai_train.config import Run
from gouai_train.targets import ResidualSpec
from gouai_train import features as F
from gouai_train import plots

STEP = '03_fit_trends'


def main():
    run = Run.from_args(__doc__)
    cfg = run.cfg
    run.require('splits/splits.json', '02_make_splits.py')
    tab = pd.read_csv(run.path('data', 'event_table.csv'))
    split = run.read_json('splits/splits.json')
    S, nch, pt = tab[f'S_{cfg["input"]}'].values, tab.nch.values, tab.pt.values
    deg = cfg['targets']['poly_degree']
    msg = []
    for f in split['folds']:
        ii = np.array(f['train'])
        spec = ResidualSpec.fit(S[ii], nch[ii], pt[ii], deg)
        run.write_json(f'targets/fold_{f["fold"]}_spec.json', spec.to_dict())
        msg.append(f'fold {f["fold"]}: σ_N={spec.sigma["nch"]:.4f} σ_p={spec.sigma["pt"]:.4f}')

    tr = np.array(split['outer_train'])
    spec = ResidualSpec.fit(S[tr], nch[tr], pt[tr], deg)
    run.write_json('targets/final_spec.json', spec.to_dict())
    run.write_json('targets/gref.json', spec.to_dict())

    z = spec.transform(S, nch, pt)
    n2, p2 = spec.inverse(S, z)
    assert np.allclose(n2, nch, rtol=1e-9) and np.allclose(p2, pt, rtol=1e-9), 'target round trip failed'

    names = cfg['features']['scalars']
    norms = F.scalar_norms(names, tab.iloc[tr], cfg['input'])
    run.write_json('targets/meta.json', {
        'channels': cfg['features']['channels'], 'scalars': names, 'scalar_norms': norms,
        'max_S_outer_train': float(S[tr].max()),
        'S_range_outer_train': [float(S[tr].min()), float(S[tr].max())],
        'image': ' ; '.join(cfg['features']['channels']) + ' (features.py)',
        'n_outer_train': int(len(tr))})
    lg = np.log(nch[tr])
    frac = 1 - np.var(lg - spec.trends['nch'].log_g(S[tr])) / np.var(lg)
    plots.trend_fit(spec, S[tr], {'nch': nch[tr], 'pt': pt[tr]}, run.path('targets', 'trend_fit'),
                    f'{cfg["label"]}: g(S) on {len(tr)} training events (degree {deg})')
    run.log(STEP, f'g(S) explains {100 * frac:.2f}% of var(log N_ch); final σ_N={spec.sigma["nch"]:.4f} '
                  f'σ_p={spec.sigma["pt"]:.4f}; ' + '; '.join(msg) + '; round trip OK')


if __name__ == '__main__':
    main()
