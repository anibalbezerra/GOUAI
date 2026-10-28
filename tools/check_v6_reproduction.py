#!/usr/bin/env python3
"""
check_v6_reproduction.py - prove that this code feeds the published v6 networks exactly as in training.

    python tools/check_v6_reproduction.py --config configs/examples/reproduce_v6_PbPb.yaml
    python tools/check_v6_reproduction.py --config ... --install-as-cv

Needs steps 01-03 of a reproduction run (original data + published split) and the v6 weights
(scripts/fetch_v6_weights.py). No GPU needed (a few minutes on a CPU).

For each fold k, the published network k is loaded into this repository's network, and its
out-of-fold and held-out predictions are computed with this repository's data path (step 01 grids,
features.py channels, targets.py specs from step 03). They must agree with the published
predictions (reference/v6/<system>/<input>_pt/*_predictions.csv) to better than 1e-3.

Why not exactly: the published predictions were computed on RTX 6000 Ada GPUs, where TensorFlow
uses TF32 arithmetic for convolutions by default; on a CPU (full float32) the network output z differs
by ~1e-3, i.e. the prediction by ~1e-4. On the SAME device this code and the original v6 code agree
to ~3e-7 (checked on 2026-10-28). A real mistake (wrong normalization, channel or spec) gives
deviations of percent or more.

--install-as-cv  also writes the results as if step 05 had trained these networks
                 (<run_dir>/cv/fold_<k>/), so that steps 06, 07 and 08 can be practised, and checked,
                 without any GPU training. Do NOT use it in a run whose own networks you want to keep.
"""
import os
import shutil

import numpy as np
import pandas as pd

import _tools_path  # noqa: F401
from gouai_train.config import Run, read_json
from gouai_train.store import EventStore
from gouai_train.targets import ResidualSpec
from gouai_train.centrality import CentralityTable
from gouai_train import metrics as MT

STEP = 'check_v6_reproduction'


def main():
    run = Run.from_args(__doc__, extra=lambda ap: ap.add_argument('--install-as-cv', action='store_true'))
    cfg = run.cfg
    run.require('targets/meta.json', '03_fit_trends.py')
    from gouai_train import model as M, training as T
    vdir = os.path.join(cfg['reference']['models_dir'], cfg['system'], f'{cfg["input"]}_pt')
    split = run.read_json('splits/splits.json')
    vsplit = read_json(os.path.join(cfg['reference']['models_dir'], cfg['system'], 'splits.json'))
    assert split['outer_test'] == vsplit['outer_test'] and split['folds'] == vsplit['folds'], \
        'this run does not use the published split (splits.import_from: reference/v6/<system>/splits.json)'
    store = EventStore(run)
    pub_oof = pd.read_csv(os.path.join(vdir, 'train_oof_predictions.csv')).set_index('idx_raw')
    pub_test = pd.read_csv(os.path.join(vdir, 'heldout_predictions.csv')).set_index('idx_raw')
    gref = ResidualSpec.from_dict(run.read_json('targets/gref.json'))
    cent = CentralityTable(cfg['evaluation']['centrality_table'])
    norms = run.read_json('targets/meta.json')['scalar_norms']
    worst, test_logs = 0.0, []
    for f in split['folds']:
        k = f['fold']
        fdir = os.path.join(vdir, f'fold_{k}')
        spec_pub = read_json(os.path.join(fdir, 'spec.json'))
        spec = ResidualSpec.from_dict(run.read_json(f'targets/fold_{k}_spec.json'))
        net = M.build_for_run(cfg)
        net.load_weights(os.path.join(fdir, 'model.weights.h5'))
        df = pd.concat([T.prediction_table(net, cfg, store, spec, norms, f['val'], 'oof'),
                        T.prediction_table(net, cfg, store, spec, norms, split['outer_test'], 'test')])
        o = df[df.part == 'oof'].set_index('idx')
        dev = float(np.max(np.abs(o.nch_pred / pub_oof.loc[o.index, 'nch_nn'] - 1)))
        dev = max(dev, float(np.max(np.abs(o.pt_pred / pub_oof.loc[o.index, 'pt_nn_GeV'] - 1))))
        worst = max(worst, dev)
        t = df[df.part == 'test'].set_index('idx').loc[split['outer_test']]
        test_logs.append(np.stack([np.log(t.nch_pred.values), np.log(t.pt_pred.values)], 1))
        print(f'fold {k}: out-of-fold predictions agree with the published ones to {dev:.1e}', flush=True)
        if run.args.install_as_cv:
            out = run.subdir('cv', f'fold_{k}')
            shutil.copyfile(os.path.join(fdir, 'model.weights.h5'), os.path.join(out, 'best.weights.h5'))
            run.write_json(f'cv/fold_{k}/spec.json', spec_pub)
            shutil.copyfile(os.path.join(fdir, 'history.csv'), os.path.join(out, 'history.csv'))
            df.to_csv(os.path.join(out, 'predictions.csv'), index=False)
            mets = {}
            for part in ('oof', 'test'):
                p = df[df.part == part]
                mets[part] = MT.evaluate(p.S.values, {'nch': p.nch_true.values, 'pt': p.pt_true.values},
                                         {'nch': p.nch_pred.values, 'pt': p.pt_pred.values}, gref, cent(p.S.values), cfg)
            run.write_json(f'cv/fold_{k}/metrics.json', {'info': read_json(os.path.join(fdir, 'metrics.json'))['info'],
                                                         'installed_from': fdir, **mets})
    ens = np.exp(np.mean(test_logs, axis=0))
    p = pub_test.loc[split['outer_test']]
    dev_e = max(float(np.max(np.abs(ens[:, 0] / p.nch_nn - 1))), float(np.max(np.abs(ens[:, 1] / p.pt_nn_GeV - 1))))
    ok = max(worst, dev_e) < 1e-3
    run.log(STEP, f'{"OK" if ok else "MISMATCH"}: out-of-fold predictions reproduce the published ones to {worst:.1e}, '
                  f'held-out ensemble to {dev_e:.1e}' + (' ; published networks installed as cv/' if run.args.install_as_cv else ''))
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
