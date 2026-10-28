"""
store.py
--------
EventStore: read access to the prepared data of a run (output of step 01), without TensorFlow.

    store = EventStore(run)
    store.table                     DataFrame, one row per event (idx, event_id, nch, pt, S_entropy, ...)
    store.S(idx)                    total S of the input density
    store.images(idx)               (n, 280, 280, n_channels) float32 network images
    store.scalars(idx, norms)       (n, n_scalars) float32 scalar inputs
    store.grids_sums(idx)           raw grids and pixel sums (used to feed other models, e.g. v6)

The grids are memory-mapped from data/grid_<density>.npy, so opening a store is instant and only
the events you ask for are read from disk.
"""
import os

import numpy as np
import pandas as pd

from . import features as F


class EventStore:
    def __init__(self, run):
        cfg = run.cfg
        self.input = cfg['input']
        self.channels = cfg['features']['channels']
        self.scalar_names = cfg['features']['scalars']
        self.table = pd.read_csv(os.path.join(run.dir, 'data', 'event_table.csv'))
        self.grids = {}
        for d in F.densities_needed(self.channels) + [self.input]:
            p = os.path.join(run.dir, 'data', f'grid_{d}.npy')
            if not os.path.exists(p):
                raise FileNotFoundError(f'{p} missing: channel(s) need the {d} density '
                                        f'(is data.x_kind energy? re-run step 01 after changing channels)')
            self.grids[d] = np.load(p, mmap_mode='r')
        self.n = len(self.table)
        assert all(len(g) == self.n for g in self.grids.values()), 'grids and event table differ in length'

    def S(self, idx=None):
        s = self.table[f'S_{self.input}'].values
        return s if idx is None else s[np.asarray(idx)]

    def y(self, idx):
        idx = np.asarray(idx)
        return {'nch': self.table.nch.values[idx], 'pt': self.table.pt.values[idx]}

    def grids_sums(self, idx, densities=None):
        idx = np.asarray(idx) if idx is not None else np.arange(self.n)
        densities = densities or list(self.grids)
        g = {d: np.asarray(self.grids[d][idx]) for d in densities}
        s = {d: self.table[f'S_{d}'].values[idx] for d in densities}
        return g, s

    def images(self, idx):
        """Network images for the events idx (in the order given)."""
        idx = np.asarray(idx)
        order = np.argsort(idx)                       # memmap reads are fastest in increasing order
        g, s = self.grids_sums(idx[order], F.densities_needed(self.channels))
        img = F.make_images(g, s, self.channels)
        out = np.empty_like(img)
        out[order] = img
        return out

    def scalars(self, idx, norms):
        return F.make_scalars(self.scalar_names, norms, self.table.iloc[np.asarray(idx)], self.input)
