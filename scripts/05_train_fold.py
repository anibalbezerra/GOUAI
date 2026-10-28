#!/usr/bin/env python3
"""
STEP 05 - train one cross-validation network                 (1 GPU per fold; sbatch slurm/train_cv.sbatch)

    python scripts/05_train_fold.py --config configs/my_experiment.yaml --fold 0      (... 1, 2, 3, 4)

Network k is trained on the training events of fold k, stopped early on its validation events, and
then predicts:
    its validation events  -> "out-of-fold" (OOF) predictions: honest predictions for training-sample
                              events, used by step 06 (Gate 2)
    the held-out test set  -> used ONLY by step 07

Outputs in <run_dir>/cv/fold_<k>/:
    best.weights.h5     the weights of the best epoch after the warm-up
    spec.json           the trend/normalization of this network (copy of targets/fold_<k>_spec.json)
    history.csv         loss per epoch (training and validation)
    predictions.csv     one row per OOF and test event (part = oof | test)
    metrics.json        training info (best epoch, time) and OOF / test metrics

A finished fold is not retrained (use --force to retrain it).
"""
import os

import numpy as np
import pandas as pd

import _path  # noqa: F401
from gouai_train.config import Run
from gouai_train.store import EventStore
from gouai_train.targets import ResidualSpec
from gouai_train.centrality import CentralityTable
from gouai_train import training as T
from gouai_train import metrics as MT

STEP = '05_train_fold'


def extra(ap):
    ap.add_argument('--fold', type=int, default=None, help='fold number 0..n_folds-1 (default: $SLURM_ARRAY_TASK_ID)')
    ap.add_argument('--force', action='store_true', help='retrain even if this fold is finished')


def main():
    run = Run.from_args(__doc__, extra)
    cfg = run.cfg
    k = run.args.fold if run.args.fold is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID', -1))
    if not 0 <= k < cfg['splits']['n_folds']:
        raise SystemExit('give --fold 0..4 (or run as a Slurm array task)')
    step = f'{STEP} fold {k}'
    out_dir = run.subdir('cv', f'fold_{k}')
    if os.path.exists(os.path.join(out_dir, 'metrics.json')) and not run.args.force:
        run.log(step, 'already finished (metrics.json exists); use --force to retrain')
        return
    run.require('targets/meta.json', '03_fit_trends.py')
    T.require_gpu(run, step)
    split = run.read_json('splits/splits.json')
    f = split['folds'][k]
    test = split['outer_test']
    assert not (set(f['train']) | set(f['val'])) & set(test), 'fold events in the held-out test set!'

    store = EventStore(run)
    spec = ResidualSpec.from_dict(run.read_json(f'targets/fold_{k}_spec.json'))
    run.write_json(f'cv/fold_{k}/spec.json', spec.to_dict())
    norms = run.read_json('targets/meta.json')['scalar_norms']
    run.log(step, f'started: {len(f["train"])} training, {len(f["val"])} validation events')

    model, info = T.fit_network(run, store, spec, norms, f['train'], f['val'], out_dir,
                                cfg['training']['epochs'], seed=cfg['training']['seed'] + 100 * k)
    df = pd.concat([T.prediction_table(model, cfg, store, spec, norms, f['val'], 'oof'),
                    T.prediction_table(model, cfg, store, spec, norms, test, 'test')])
    df.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)

    gref = ResidualSpec.from_dict(run.read_json('targets/gref.json'))
    cent = CentralityTable(cfg['evaluation']['centrality_table'])
    mets = {}
    for part in ('oof', 'test'):
        p = df[df.part == part]
        mets[part] = MT.evaluate(p.S.values, {'nch': p.nch_true.values, 'pt': p.pt_true.values},
                                 {'nch': p.nch_pred.values, 'pt': p.pt_pred.values}, gref, cent(p.S.values), cfg)
    run.write_json(f'cv/fold_{k}/metrics.json', {'info': info, **mets})
    m = mets['oof']
    run.log(step, f'best epoch {info["best_epoch"]}/{info["epochs_run"]}, {info["wall_s"] / 3600:.2f} h; '
                  f'OOF: N_ch rel. err {100 * m["nch"]["all"]["mean_rel_err"]:.2f}%, '
                  f'β_N={m["nch"]["nch_ge_floor"]["beta_cal"]:.2f}, err/σ_N={m["nch"]["nch_ge_floor"]["err_ratio"]:.2f}, '
                  f'[pT] rel. err {100 * m["pt"]["all"]["mean_rel_err"]:.2f}%')


if __name__ == '__main__':
    main()
