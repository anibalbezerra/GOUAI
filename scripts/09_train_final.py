#!/usr/bin/env python3
"""
STEP 09 (optional) - one network on all training events          (1 GPU; sbatch slurm/gpu_step.sbatch)

    python scripts/09_train_final.py --config configs/my_experiment.yaml

Needs step 06 (it writes cv/final_plan.json). Trains one network on ALL outer-train events, for a fixed
number of epochs = the median best epoch of the 5 folds (there is no validation set left to stop on),
and predicts the held-out test set. Outputs in <run_dir>/final/.

The published v6 predictions use the 5-fold ENSEMBLE, which was more accurate than this single network
in both systems. Train it only if you want to repeat that comparison; afterwards re-run step 07, which
then reports both.
"""
import os

import _path  # noqa: F401
from gouai_train.config import Run
from gouai_train.store import EventStore
from gouai_train.targets import ResidualSpec
from gouai_train import training as T

STEP = '09_train_final'


def main():
    run = Run.from_args(__doc__)
    cfg = run.cfg
    run.require('cv/final_plan.json', '06_cv_report.py')
    T.require_gpu(run, STEP)
    split = run.read_json('splits/splits.json')
    plan = run.read_json('cv/final_plan.json')
    store = EventStore(run)
    spec = ResidualSpec.from_dict(run.read_json('targets/final_spec.json'))
    norms = run.read_json('targets/meta.json')['scalar_norms']
    out_dir = run.subdir('final')
    run.write_json('final/spec.json', spec.to_dict())
    run.log(STEP, f'started: {len(split["outer_train"])} events, {plan["epochs"]} epochs')
    model, info = T.fit_network(run, store, spec, norms, split['outer_train'], None, out_dir,
                                plan['epochs'], seed=cfg['training']['seed'] + 999)
    df = T.prediction_table(model, cfg, store, spec, norms, split['outer_test'], 'test')
    df.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)
    run.write_json('final/metrics.json', {'info': info, 'plan': plan})
    run.log(STEP, f'{info["epochs_run"]} epochs in {info["wall_s"] / 3600:.2f} h; now re-run step 07')


if __name__ == '__main__':
    main()
