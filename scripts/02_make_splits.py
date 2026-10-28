#!/usr/bin/env python3
"""
STEP 02 - choose the held-out test events and the cross-validation folds      (CPU, seconds)

    python scripts/02_make_splits.py --config configs/my_experiment.yaml

Writes:
    <run_dir>/splits/splits.json        outer_train, outer_test, folds[k].train / .val  (indices = event_table idx)
    <run_dir>/splits/splits_report.md   how many events of each centrality class are in train / test

Rules (see gouai_train/splits.py):
  * splits.test_fraction (25 %) of the events are held out, stratified on S so that every
    centrality, including the ultra-central events, is represented in the test set;
  * v6 held-out events found in this sample are forced into the test set (splits.protect_reference_test);
  * the remaining events are divided into 5 folds, also stratified on S;
  * splits.import_from: reuse an existing splits.json verbatim (for example the v6 split, to reproduce
    v6, or the split of your full run, for learning-curve runs);
  * splits.train_subsample: train on a subset only (the test set stays the same).

The split is written ONCE. Re-running this step refuses to overwrite it (use --force only if no network
has been trained with it yet: changing the test set after looking at test results is not allowed).
"""
import os

import numpy as np
import pandas as pd

import _path  # noqa: F401
from gouai_train.config import Run, read_json
from gouai_train import splits as SP
from gouai_train.centrality import class_index, class_labels

STEP = '02_make_splits'


def main():
    run = Run.from_args(__doc__, extra=lambda ap: ap.add_argument('--force', action='store_true'))
    cfg, sc = run.cfg, run.cfg['splits']
    run.require('data/event_table.csv', '01_prepare_data.py')
    out = os.path.join(run.dir, 'splits', 'splits.json')
    if os.path.exists(out) and not run.args.force:
        raise SystemExit(f'{out} exists. The split must not change once networks are trained with it. '
                         'Delete it by hand (or use --force) only if you are sure.')
    tab = pd.read_csv(run.path('data', 'event_table.csv'))
    n = len(tab)
    S = tab[f'S_{cfg["input"]}'].values
    notes = []

    if sc.get('import_from'):
        split = read_json(sc['import_from'])
        split = {k: split[k] for k in ('outer_train', 'outer_test', 'folds')}
        SP.check_split(split, n)
        covered = set(split['outer_train']) | set(split['outer_test'])
        if covered != set(range(n)) and not sc.get('train_subsample'):
            notes.append(f'imported split covers {len(covered)} of {n} events; the others are unused')
        notes.append(f'split imported verbatim from `{sc["import_from"]}`')
        tr, te = np.array(split['outer_train']), np.array(split['outer_test'])
    else:
        forced = []
        if sc['protect_reference_test'] and 'v6_part' in tab:
            forced = tab.idx[tab.v6_part == 'test'].values
            notes.append(f'{len(forced)} v6 held-out events forced into the test set')
        elif sc['protect_reference_test']:
            notes.append('protect_reference_test: overlap with v6 unknown (reference.old_data.y_file not set in step 01)')
        tr, te = SP.outer_split(S, sc['test_fraction'], sc['seed'], sc['n_strata'], forced)
        split = None

    if sc.get('train_subsample'):
        full = len(tr)
        tr = SP.subsample(tr, S, int(sc['train_subsample']), sc['subsample_seed'], sc['n_strata'])
        notes.append(f'training events subsampled: {len(tr)} of {full} (seed {sc["subsample_seed"]}); test set unchanged')
        split = None
    if split is None:
        split = {'outer_train': tr.tolist(), 'outer_test': te.tolist(),
                 'folds': SP.make_folds(tr, S, sc['n_folds'], sc['fold_seed'], sc['n_strata'])}
    SP.check_split(split, n)
    if 'v6_part' in tab:
        leak = np.intersect1d(split['outer_train'], tab.idx[tab.v6_part == 'test'].values)
        if len(leak):
            notes.append(f'**WARNING** {len(leak)} v6 held-out events are in the TRAINING set: the comparison '
                         'on the v6 test set (step 08) will exclude them')
    split['info'] = {'n_events': n, 'n_train': len(split['outer_train']), 'n_test': len(split['outer_test']),
                     'notes': notes, 'config': sc}
    run.write_json('splits/splits.json', split)

    edges = cfg['evaluation']['centrality_edges']
    k = class_index(tab.centrality.values, edges)
    in_tr, in_te = np.isin(tab.idx, split['outer_train']), np.isin(tab.idx, split['outer_test'])
    lines = [f'# Splits: {cfg["label"]}\n', *[f'- {x}' for x in notes],
             f'- outer train **{len(split["outer_train"])}**, held-out test **{len(split["outer_test"])}**, '
             f'{sc["n_folds"]} folds of ~{len(split["outer_train"]) // sc["n_folds"]} validation events\n',
             '| centrality class | train | test |', '|---|---|---|']
    lines += [f'| {lab} | {int(np.sum(in_tr & (k == j)))} | {int(np.sum(in_te & (k == j)))} |'
              for j, lab in enumerate(class_labels(edges))]
    with open(run.path('splits', 'splits_report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    run.log(STEP, f'train {len(split["outer_train"])}, test {len(split["outer_test"])}; ' + '; '.join(notes))


if __name__ == '__main__':
    main()
