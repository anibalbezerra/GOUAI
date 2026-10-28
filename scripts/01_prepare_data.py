#!/usr/bin/env python3
"""
STEP 01 - prepare the data                                    (CPU, minutes; no GPU needed)

    python scripts/01_prepare_data.py --config configs/my_experiment.yaml

Reads the raw hybrid events (data.x_path, data.y_file), checks them, and writes:

    <run_dir>/data/event_table.csv     one row per usable event:
                                         idx        row number used everywhere else (0 ... N-1)
                                         event_id   the id from the label file
                                         nch, pt    the labels (pT in GeV)
                                         S_entropy  total entropy (pixel sum of the entropy grid)
                                         S_energy   total energy  (only if the raw files are energy density)
                                         centrality minimum-bias centrality (%) from S of the input density
                                         v6_idx, v6_part   if the event is also in the v6 sample:
                                                    its row there and whether v6 used it to train or test
                                         + any data.extra_y_columns
    <run_dir>/data/grid_entropy.npy    (N, 280, 280) float32 entropy density
    <run_dir>/data/grid_energy.npy     (N, 280, 280) float32 energy density (if available)
    <run_dir>/data/excluded_events.csv events that could not be used, and why
    <run_dir>/data/data_report.md      counts, ranges and centrality coverage   <- READ THIS
    <run_dir>/data/data_overview.png   events per centrality class (this sample vs v6), N_ch(S), [pT](S)

Energy -> entropy: if the raw files are energy density (data.x_kind: energy), the entropy density is
computed with the same equation of state and the same cubic interpolation as for the v6 training
images (gouai_train/eos.py). S is always summed in double precision from the raw values.
"""
import os
import time

import numpy as np
import pandas as pd

import _path  # noqa: F401
from gouai_train.config import Run, read_json
from gouai_train import rawdata, eos, splits
from gouai_train import features as F
from gouai_train.centrality import CentralityTable, class_labels
from gouai_train import plots

STEP = '01_prepare_data'


def reference_overlap(cfg, labels):
    """Match the new events to the v6 sample by their labels; returns (v6_idx, v6_part) arrays or None."""
    old = cfg['reference']['old_data']
    if not old.get('y_file'):
        return None, 'reference.old_data.y_file not set: overlap with the v6 sample not checked'
    ycfg = {'data': {'y_file': old['y_file'], 'y_columns': {'id': 0, 'nch': 1, 'pt': 2}, 'extra_y_columns': {}}}
    ref = rawdata.read_labels(ycfg)
    split = read_json(os.path.join(cfg['reference']['models_dir'], cfg['system'], 'splits.json'))
    part = np.array([''] * len(ref), dtype=object)
    part[split['outer_train']] = 'train'
    part[split['outer_test']] = 'test'
    m = splits.match_events(labels.nch.values, labels.pt.values, ref.nch.values, ref.pt.values)
    v6_part = np.where(m >= 0, part[np.maximum(m, 0)], '')
    n_tr, n_te = int(np.sum(v6_part == 'train')), int(np.sum(v6_part == 'test'))
    return (m, v6_part), (f'{int((m >= 0).sum())} events are also in the v6 sample '
                          f'({n_tr} were v6 training events, {n_te} v6 held-out events)')


def v6_sample_centrality(cfg, cent_table):
    """Centralities of the v6 training sample, from the published prediction tables (no raw data needed)."""
    d = os.path.join(cfg['reference']['models_dir'], cfg['system'], f'{cfg["input"]}_pt')
    parts = [pd.read_csv(os.path.join(d, f)) for f in ('train_oof_predictions.csv', 'heldout_predictions.csv')
             if os.path.exists(os.path.join(d, f))]
    if not parts:
        return None
    t = pd.concat(parts)
    col = f'S_{cfg["input"]}'
    return cent_table(t[col].values) if col in t else None


def main():
    run = Run.from_args(__doc__)
    cfg = run.cfg
    d = cfg['data']
    run.save_config()

    if d.get('reuse_prepared_from'):
        src = os.path.join(os.path.abspath(d['reuse_prepared_from']), 'data')
        dst = os.path.join(run.dir, 'data')
        if not os.path.exists(os.path.join(src, 'event_table.csv')):
            raise SystemExit(f'{src} has no prepared data (run step 01 there first)')
        if not os.path.exists(dst):
            os.symlink(src, dst)
        run.log(STEP, f'data/ linked to {src} (no raw files read)')
        return

    run.log(STEP, f'started: x_path={d["x_path"]} (x_kind {d["x_kind"]}), y_file={d["y_file"]}')
    labels = rawdata.read_labels(cfg)
    has, notes = rawdata.match_report(cfg, labels)
    n = int(has.sum())
    densities = ['entropy', 'energy'] if d['x_kind'] == 'energy' else ['entropy']
    g = d['grid_n']
    tmp = {k: np.lib.format.open_memmap(run.path('data', f'grid_{k}.npy.tmp'), mode='w+',
                                        dtype=np.float32, shape=(n, g, g)) for k in densities}
    sums = {k: np.zeros(n) for k in densities}
    derived = F.derived_needed(cfg)
    dcols = {name: np.zeros(n) for name in derived}
    rows = np.zeros(n, dtype=int)
    bad_grid = []
    t0 = time.time()
    for i, (row, x) in enumerate(rawdata.iter_events(cfg, labels)):
        rows[i] = row
        if not np.all(np.isfinite(x)) or x.min() < 0:
            bad_grid.append((row, 'grid has NaN/inf or negative values'))
        grids = {'energy': x, 'entropy': eos.entropy_density(x, d['eos_file'])} if d['x_kind'] == 'energy' \
            else {'entropy': x}
        for k in densities:
            sums[k][i] = grids[k].sum()
            tmp[k][i] = grids[k].reshape(g, g)
        for name in derived:
            dcols[name][i] = F.DERIVED[name]({k: v.reshape(1, g, g) for k, v in grids.items()})[0]
        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f'  {i + 1}/{n} events read ({time.time() - t0:.0f} s)', flush=True)
    for a in tmp.values():
        a.flush()
    del tmp
    tmp = {k: np.load(run.path('data', f'grid_{k}.npy.tmp'), mmap_mode='r') for k in densities}

    # -------------------------------------------------------------- table and quality cuts
    tab = labels.iloc[rows].reset_index(drop=True)
    for k in densities:
        tab[f'S_{k}'] = sums[k]
    for name in derived:
        tab[name] = dcols[name]
    reason = pd.Series([''] * n, dtype=object)
    if d.get('extra_table'):
        extra = pd.read_csv(d['extra_table'])
        if 'event_id' not in extra:
            raise SystemExit(f'{d["extra_table"]} needs an event_id column')
        if extra.event_id.duplicated().any():
            raise SystemExit(f'{d["extra_table"]}: event_id values are not unique')
        new_cols = [c for c in extra.columns if c != 'event_id' and c not in tab]
        tab = tab.merge(extra[['event_id'] + new_cols], on='event_id', how='left')
        reason[tab[new_cols].isna().any(axis=1).values] = 'missing in data.extra_table'
        notes.append(f'extra_table: columns {new_cols} merged by event_id')
    for row, why in bad_grid:
        reason[np.where(rows == row)[0]] = why
    lab_ok = np.isfinite(tab.nch) & np.isfinite(tab.pt)
    reason[~lab_ok] = 'label is NaN/inf'
    reason[lab_ok & (tab.nch <= 0)] = 'N_ch <= 0 (log of the target undefined)'
    reason[lab_ok & (tab.pt <= 0)] = '[pT] <= 0'
    reason[tab[f'S_{cfg["input"]}'] <= 0] = 'total S = 0 (empty grid)'
    keep = (reason == '').values
    excluded = pd.concat([tab[~keep].assign(reason=reason[~keep].values),
                          labels[~has].assign(reason='no grid file')], ignore_index=True)
    excluded.to_csv(run.path('data', 'excluded_events.csv'), index=False)

    tab = tab[keep].reset_index(drop=True)
    for k in densities:
        final = np.lib.format.open_memmap(run.path('data', f'grid_{k}.npy'), mode='w+',
                                          dtype=np.float32, shape=(len(tab), g, g))
        kept = np.where(keep)[0]
        for i in range(0, len(kept), 500):                      # in chunks: bounded memory
            final[i:i + 500] = tmp[k][kept[i:i + 500]]
        final.flush()
        del final
        os.remove(run.path('data', f'grid_{k}.npy.tmp'))
    tab.insert(0, 'idx', np.arange(len(tab)))

    cent_table = CentralityTable(cfg['evaluation']['centrality_table'])
    tab['centrality'] = cent_table(tab[f'S_{cfg["input"]}'].values)
    ov, ov_note = reference_overlap(cfg, tab)
    notes.append(ov_note)
    if ov is not None:
        tab['v6_idx'], tab['v6_part'] = ov[0], ov[1]
    tab.to_csv(run.path('data', 'event_table.csv'), index=False, float_format='%.10g')

    # -------------------------------------------------------------- report
    warn = []
    dup = tab.duplicated(subset=['nch', 'pt'], keep=False)
    if dup.any():
        warn.append(f'{int(dup.sum())} events share identical (N_ch, [pT]): duplicated events? check event_table.csv')
    odd_pt = (tab.pt < 0.2) | (tab.pt > 2.0)
    if odd_pt.any():
        warn.append(f'{int(odd_pt.sum())} events have [pT] outside 0.2-2 GeV: is column {d["y_columns"]["pt"]} really [pT] in GeV?')
    edges = cfg['evaluation']['centrality_edges']
    S_in = tab[f'S_{cfg["input"]}']
    cent_v6 = v6_sample_centrality(cfg, cent_table)
    h_new = np.histogram(tab.centrality, edges)[0]
    h_v6 = np.histogram(cent_v6, edges)[0] if cent_v6 is not None else [None] * len(h_new)
    lines = [f'# Data report: {cfg["label"]}, input {cfg["input"]}\n',
             f'- usable events: **{len(tab)}** (excluded: {len(excluded)}, see `excluded_events.csv`)',
             *[f'- {x}' for x in notes],
             f'- N_ch range [{tab.nch.min():.3g}, {tab.nch.max():.4g}], [pT] range [{tab.pt.min():.4f}, {tab.pt.max():.4f}] GeV',
             f'- S_{cfg["input"]} range [{S_in.min():.4g}, {S_in.max():.4g}]',
             f'- centrality table: `{cfg["evaluation"]["centrality_table"]}`',
             *[f'- **WARNING** {w}' for w in warn],
             '\n## Events per minimum-bias centrality class\n',
             '| class | this sample | v6 sample |', '|---|---|---|']
    lines += [f'| {lab} | {a} | {"–" if b is None else b} |' for lab, a, b in zip(class_labels(edges), h_new, h_v6)]
    with open(run.path('data', 'data_report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    plots.data_overview(tab.centrality.values, tab[f'S_{cfg["input"]}'].values, tab.nch.values, tab.pt.values,
                        edges, run.path('data', 'data_overview'), cent_ref=cent_v6,
                        title=f'{cfg["label"]}: {len(tab)} events')
    run.log(STEP, f'{len(tab)} events ready ({len(excluded)} excluded); '
                  f'{int(h_new[0])} in the first class {class_labels(edges)[0]}; {ov_note}. '
                  f'Read data/data_report.md')


if __name__ == '__main__':
    main()
