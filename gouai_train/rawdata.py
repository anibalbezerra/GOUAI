"""
rawdata.py
----------
Reading the hybrid-model events as they come from the simulations, in either of two layouts.

Layout A, one file with all events (the format of the original training sample):
    x_path = .../x_energy.dat     line i = event i: 78400 numbers (the 280 x 280 grid, row by row)
    y_file = .../Y_eta0p5_pt.dat  line i = the labels of the SAME event i

Layout B, one file per event (the format the simulations produce):
    x_path = .../xenergy/         files such as flattened_IC_17.txt, one per event; the number in the
                                  file name (first group of data.x_pattern) is the event id.
                                  Each file holds the 78400 numbers, as one column or as a 280 x 280 table.
    y_file = .../Y_eta0p5_pt.dat  column data.y_columns.id holds the same event id.

In layout A events are matched by line number, in layout B by event id. Layout B is safer:
a missing or extra file cannot shift all labels by one event.

Labels (y_file): whitespace-separated numbers; by default column 0 = event id, column 1 = N_ch,
column 2 = [pT] in GeV (charged particles, pT >= 0.15 GeV, |eta| < 0.5).
"""
import os
import re

import numpy as np
import pandas as pd


def read_labels(cfg):
    """DataFrame with event_id, nch, pt and any data.extra_y_columns, in file order."""
    d = cfg['data']
    y = pd.read_csv(d['y_file'], sep=r'\s+', header=None, comment='#').values
    cols = d['y_columns']
    out = pd.DataFrame({'event_id': y[:, cols['id']].astype(np.int64),
                        'nch': y[:, cols['nch']].astype(np.float64),
                        'pt': y[:, cols['pt']].astype(np.float64)})
    for name, c in (d.get('extra_y_columns') or {}).items():
        out[name] = y[:, c].astype(np.float64)
    return out


def read_grid_file(path, n_pixels):
    """One event file (layout B) -> flat float64 array of n_pixels values."""
    a = pd.read_csv(path, sep=r'\s+', header=None, comment='#', engine='c').values.ravel()
    if a.size != n_pixels:
        raise ValueError(f'{path}: {a.size} numbers, expected {n_pixels}')
    return a.astype(np.float64)


def list_event_files(x_dir, pattern):
    """{event_id: path} for the files in x_dir whose name matches `pattern` (first group = id)."""
    rx = re.compile(pattern)
    files = {}
    for name in os.listdir(x_dir):
        m = rx.search(name)
        if m is None:
            continue
        eid = int(m.group(1))
        if eid in files:
            raise ValueError(f'two files with event id {eid}: {files[eid]} and {name}')
        files[eid] = os.path.join(x_dir, name)
    return files


def iter_events(cfg, labels):
    """
    Yield (row, flat_grid) for every label row that has a density grid, in label order.
    `row` is the position in `labels`. Rows without a grid are skipped silently here;
    `match_report` lists them.
    """
    d = cfg['data']
    n_pix = d['grid_n'] ** 2
    if os.path.isdir(d['x_path']):
        files = list_event_files(d['x_path'], d['x_pattern'])
        for row, eid in enumerate(labels.event_id.values):
            if eid in files:
                yield row, read_grid_file(files[eid], n_pix)
    else:
        with open(d['x_path']) as f:
            for row, line in enumerate(f):
                a = np.array(line.split(), dtype=np.float64)
                if a.size != n_pix:
                    raise ValueError(f'{d["x_path"]} line {row + 1}: {a.size} numbers, expected {n_pix}')
                yield row, a


def match_report(cfg, labels):
    """
    Which label rows have a grid and which grids have no label.
    Returns (has_grid: bool array over label rows, notes: list of strings).
    """
    d = cfg['data']
    notes = []
    if os.path.isdir(d['x_path']):
        files = list_event_files(d['x_path'], d['x_pattern'])
        ids = labels.event_id.values
        has = np.isin(ids, list(files))
        orphans = sorted(set(files) - set(ids.tolist()))
        notes.append(f'layout B (one file per event): {len(files)} grid files, {len(ids)} label rows, '
                     f'{int(has.sum())} matched by event id')
        if (~has).any():
            notes.append(f'{int((~has).sum())} label rows have NO grid file (ids {ids[~has][:10].tolist()}...)')
        if orphans:
            notes.append(f'{len(orphans)} grid files have NO label row (ids {orphans[:10]}...)')
        if len(set(ids.tolist())) != len(ids):
            raise ValueError('event ids in y_file are not unique: cannot match files to labels')
    else:
        with open(d['x_path']) as f:
            n_lines = sum(1 for _ in f)
        notes.append(f'layout A (one file, one event per line): {n_lines} grid lines, {len(labels)} label rows, '
                     'matched by line number')
        if n_lines != len(labels):
            raise ValueError(f'{d["x_path"]} has {n_lines} lines but {d["y_file"]} has {len(labels)} rows: '
                             'in layout A they must correspond line by line')
        has = np.ones(len(labels), bool)
    return has, notes
