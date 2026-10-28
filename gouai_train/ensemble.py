"""
ensemble.py
-----------
Several trained networks used together. The published prediction (v6 and the default here) is the
GEOMETRIC MEAN of the 5 cross-validation networks:

    log y_ensemble = mean over folds k of log y_k ,       y_k = g_k(S) exp(mu_k + sigma_k z_k)

Each fold network carries its own trend/normalization (its spec.json), so the members are combined
in physical units, not in z. The spread of the members (std of log y_k) measures how much the
prediction depends on which events were used for training: a direct estimate of model uncertainty.

    ens = Ensemble.from_run(run)                         # the 5 networks of this experiment (step 05)
    ens = Ensemble.from_run(run, 'final')                # the single final network (step 09)
    ens = Ensemble.from_v6(models_dir, system, 'entropy')   # the published v6 networks
    nch, pt, members = ens.predict_store(store, idx)        # members: (n_networks, n_events, 2) = (log nch, log pt)
    nch, pt, members = ens.predict_grids(grids, sums)       # from raw density grids

Every ensemble knows its own channels and scalar normalization: the v6 networks use the maximum S
of THEIR training sample, the new networks that of the new sample. Using the wrong one silently
biases every prediction, so it is never taken from the run when predicting with v6.
"""
import os

import numpy as np
import pandas as pd

from . import features as F
from . import model as M
from .config import read_json
from .targets import ResidualSpec


class Ensemble:
    def __init__(self, members, channels, scalar_names, norms, input_density, name, dirs, batch_size=32):
        self.members = members              # list of (keras model, ResidualSpec)
        self.channels = channels
        self.scalar_names = scalar_names
        self.norms = norms                  # {scalar name: normalization constant}
        self.input = input_density
        self.name = name
        self.dirs = dirs
        self.batch_size = batch_size

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_run(cls, run, which='cv', folds=None):
        cfg = run.cfg
        meta = run.read_json('targets/meta.json')
        if which == 'cv':
            folds = range(cfg['splits']['n_folds']) if folds is None else folds
            dirs = [os.path.join(run.dir, 'cv', f'fold_{k}') for k in folds]
        elif which == 'final':
            dirs = [os.path.join(run.dir, 'final')]
        else:
            raise ValueError("which must be 'cv' or 'final'")
        members = []
        for d in dirs:
            w = os.path.join(d, 'best.weights.h5')
            if not os.path.exists(w):
                raise FileNotFoundError(f'{w} missing: is the training of this network finished?')
            net = M.build_for_run(cfg)
            net.load_weights(w)
            members.append((net, ResidualSpec.from_dict(read_json(os.path.join(d, 'spec.json')))))
        return cls(members, cfg['features']['channels'], cfg['features']['scalars'], meta['scalar_norms'],
                   cfg['input'], f'new ({which})', dirs, cfg['training']['predict_batch'])

    @classmethod
    def from_v6(cls, models_dir, system, input_density):
        d = os.path.join(models_dir, system, f'{input_density}_pt')
        info = read_json(os.path.join(d, 'ensemble.json'))
        members, dirs = [], []
        for k in info['folds']:
            fdir = os.path.join(d, f'fold_{k}')
            w = os.path.join(fdir, 'model.weights.h5')
            if not os.path.exists(w):
                raise FileNotFoundError(f'{w} missing: fetch the published weights first '
                                        '(python scripts/fetch_v6_weights.py, see docs/04_student_guide.md step 08)')
            net = M.build_model(info['training'])
            net.load_weights(w)
            members.append((net, ResidualSpec.from_dict(read_json(os.path.join(fdir, 'spec.json')))))
            dirs.append(fdir)
        return cls(members, [f'{input_density}_abs', f'{input_density}_norm'], ['S_norm'],
                   {'S_norm': float(info['max_S'])}, input_density, 'v6 (published)', dirs)

    # ------------------------------------------------------------------ prediction
    def _combine(self, images, scalars, S):
        logs = []
        for net, spec in self.members:
            z = M.predict_z(net, images, scalars, self.batch_size)
            nch, pt = spec.inverse(S, z)
            logs.append(np.stack([np.log(nch), np.log(pt)], axis=1))
        return np.stack(logs)                                       # (n_members, n, 2)

    def predict_grids(self, grids, sums, table=None):
        """
        grids: {density: (n, 280, 280)}, sums: {density: (n,)} raw grids and their pixel sums.
        table: DataFrame with any extra scalar columns (only needed for col:<name> scalars).
        Events with zero total input get nch = 0, pt = NaN.
        """
        S = np.asarray(sums[self.input], np.float64)
        n = len(S)
        tab = pd.DataFrame({f'S_{d}': np.asarray(v, np.float64) for d, v in sums.items()})
        if table is not None:
            for c in table.columns:
                if c not in tab:
                    tab[c] = table[c].values
        ok = S > 0
        for name in self.scalar_names:          # derived scalars (features.DERIVED) are computed from the grids
            col = name[4:] if name.startswith('col:') else None
            if col and col not in tab and col in F.DERIVED and ok.any():
                tab[col] = np.nan
                tab.loc[ok, col] = F.DERIVED[col]({d: np.asarray(g)[ok] for d, g in grids.items()})
        nch, pt = np.zeros(n), np.full(n, np.nan)
        members = np.full((len(self.members), n, 2), np.nan)
        if ok.any():
            img = F.make_images({d: np.asarray(g)[ok] for d, g in grids.items()},
                                {d: np.asarray(s)[ok] for d, s in sums.items()}, self.channels)
            sc = F.make_scalars(self.scalar_names, self.norms, tab[ok], self.input)
            logs = self._combine(img, sc, S[ok])
            members[:, ok] = logs
            mean = logs.mean(axis=0)
            nch[ok], pt[ok] = np.exp(mean[:, 0]), np.exp(mean[:, 1])
        return nch, pt, members

    def predict_store(self, store, idx, chunk=256):
        """Predictions for events idx of a run's EventStore (chunked, so any number of events fits)."""
        idx = np.asarray(idx)
        dens = sorted(set(F.densities_needed(self.channels)) | {self.input})
        out_n, out_p, out_m = [], [], []
        for i in range(0, len(idx), chunk):
            part = idx[i:i + chunk]
            g, s = store.grids_sums(part, dens)
            n_, p_, m_ = self.predict_grids(g, s, store.table.iloc[part])
            out_n.append(n_); out_p.append(p_); out_m.append(m_)
        return np.concatenate(out_n), np.concatenate(out_p), np.concatenate(out_m, axis=1)
