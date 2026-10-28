#!/usr/bin/env python3
"""
STEP 04 - smoke test                                     (1 GPU, ~5-15 minutes; sbatch slurm/gpu_step.sbatch)

    python scripts/04_smoke_test.py --config configs/my_experiment.yaml
    python scripts/04_smoke_test.py --config ... --allow-cpu --max-events 40     (quick check on a login node)

Before spending GPU-hours, check that everything fits together, on fold 0:
  1. the UNTRAINED network outputs z = 0 for every event, i.e. exactly the trend g(S) exp(mu)
     (zero-initialized heads; proves images, scalars, specs and the inverse transform are consistent);
  2. 2 epochs of real training run without errors, with a finite loss;
  3. the predictions are finite and positive.
Outputs in <run_dir>/smoke/. Nothing here is used later.
"""
import os

import numpy as np

import _path  # noqa: F401
from gouai_train.config import Run
from gouai_train.store import EventStore
from gouai_train.targets import ResidualSpec
from gouai_train import training as T
from gouai_train import model as M

STEP = '04_smoke_test'


def extra(ap):
    ap.add_argument('--allow-cpu', action='store_true', help='run without GPU (only with --max-events)')
    ap.add_argument('--max-events', type=int, default=None, help='use only this many train/val events')


def main():
    run = Run.from_args(__doc__, extra)
    cfg = run.cfg
    run.require('targets/meta.json', '03_fit_trends.py')
    T.require_gpu(run, STEP, allow_cpu=run.args.allow_cpu and run.args.max_events is not None)
    store = EventStore(run)
    f = run.read_json('splits/splits.json')['folds'][0]
    tr, va = np.array(f['train']), np.array(f['val'])
    if run.args.max_events:
        tr, va = tr[:run.args.max_events], va[:max(run.args.max_events // 4, 4)]
    spec = ResidualSpec.from_dict(run.read_json('targets/fold_0_spec.json'))
    norms = run.read_json('targets/meta.json')['scalar_norms']
    out_dir = run.subdir('smoke')

    model0 = M.build_for_run(cfg)
    nch0, pt0, z0 = T.predict_physical(model0, cfg, store, spec, norms, va)
    S = store.S(va)
    assert np.max(np.abs(z0)) == 0.0, 'zero-initialized heads do not output 0'
    assert np.allclose(nch0, spec.trends['nch'](S) * np.exp(spec.mu['nch']), rtol=1e-10)
    assert np.allclose(pt0, spec.trends['pt'](S) * np.exp(spec.mu['pt']), rtol=1e-10)
    print('check 1 OK: untrained network = trend g(S) exactly', flush=True)

    model, info = T.fit_network(run, store, spec, norms, tr, va, out_dir, cfg['smoke']['epochs'], seed=0, warmup=0)
    assert np.isfinite(info['best_val_loss']), 'non-finite validation loss'
    print('check 2 OK: training runs, validation loss', info['best_val_loss'], flush=True)
    df = T.prediction_table(model, cfg, store, spec, norms, va, 'val')
    assert np.all(np.isfinite(df[['nch_pred', 'pt_pred']].values)) and np.all(df[['nch_pred', 'pt_pred']].values > 0)
    df.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)
    run.write_json('smoke/metrics.json', {'info': info})
    sec_per_epoch = info['wall_s'] / info['epochs_run']
    est = sec_per_epoch * (len(np.array(f['train'])) / len(tr)) * 300 / 3600
    run.log(STEP, f'OK: zero-init, training and predictions; {sec_per_epoch:.0f} s/epoch on {len(tr)} events '
                  f'-> one fold of ~300 epochs on all {len(f["train"])} events takes roughly {est:.1f} h')


if __name__ == '__main__':
    main()
