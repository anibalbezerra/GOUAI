#!/usr/bin/env python3
"""
STEP 10 (optional) - use the trained ensemble on new initial conditions          (GPU)

A. Initial conditions you already have (same formats as step 01, no labels needed):

    python scripts/10_predict.py --config cfg.yaml --ic /path/x_energy.dat  --ic-kind energy
    python scripts/10_predict.py --config cfg.yaml --ic /path/xenergy_dir/  --ic-kind energy
    -> <run_dir>/predict/predictions_<name>.csv  (row, event_id, S, nch, pt, fold spread)

B. Large TRENTo samples, as for the paper's 1.5 million events per system
   (needs the trento binary and production.trento_args in the config):

    python scripts/10_predict.py --config cfg.yaml --trento-preflight         20 events, check everything
    sbatch --array=0-7 slurm/predict_trento.sbatch cfg.yaml                    production.n_tasks GPU tasks
    python scripts/10_predict.py --config cfg.yaml --trento-merge              -> predict/events.csv.gz

   Each task generates its share of production.n_events in batches of production.batch events,
   converts TRENTo's energy density exactly as in training (entropy via the EoS), predicts, and saves
   every batch (predict/parts/task_<k>/batch_<b>.npz). Finished batches are skipped, so a task that
   was killed resumes where it stopped. TRENTo seeds = seed_base + 100000 x task + batch (reproducible).

   Columns of events.csv.gz: event, b, npart, mult, e2, e3, e4, e5 (TRENTo's own summary),
   S_<input>, nch, pt, zero_input (1 = empty grid: nch = 0, pt = NaN).

The model is production.model: ensemble (default, geometric mean of the 5 fold networks) or final.
Predictions for S outside the training range are extrapolations of the trend g(S): their fraction is
reported in summary.json.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import _path  # noqa: F401
from gouai_train.config import Run, read_json, write_json
from gouai_train import eos, rawdata

STEP = '10_predict'
COLS = ['event', 'b', 'npart', 'mult', 'e2', 'e3', 'e4', 'e5']


def extra(ap):
    ap.add_argument('--ic', help='file (one event per line) or directory (one file per event)')
    ap.add_argument('--ic-kind', choices=['energy', 'entropy'], default='energy')
    ap.add_argument('--ic-pattern', default=r'(\d+)', help='directory only: regex, first group = event id')
    ap.add_argument('--trento-preflight', action='store_true')
    ap.add_argument('--trento-task', type=int, default=None)
    ap.add_argument('--trento-merge', action='store_true')


def to_grids(x, kind, cfg):
    """Raw grids (n, 78400) of `kind` -> {density: (n, 280, 280)}, {density: (n,)} for the model."""
    g = cfg['data']['grid_n']
    if kind == 'energy':
        grids = {'energy': x, 'entropy': eos.entropy_density(x, cfg['data']['eos_file'])}
    else:
        grids = {'entropy': x}
    grids = {d: a.reshape(-1, g, g) for d, a in grids.items()}
    return grids, {d: a.reshape(len(a), -1).sum(1) for d, a in grids.items()}


def load_ensemble(run):
    from gouai_train.ensemble import Ensemble
    from gouai_train import training as T
    T.require_gpu(run, STEP, allow_cpu=True)
    return Ensemble.from_run(run, 'cv' if run.cfg['production']['model'] == 'ensemble' else 'final')


# ---------------------------------------------------------------------------- A: given files
def predict_files(run):
    cfg = run.cfg
    n_pix = cfg['data']['grid_n'] ** 2
    if cfg['input'] == 'energy' and run.args.ic_kind == 'entropy':
        sys.exit('this model takes energy density; entropy files cannot be converted back')
    ens = load_ensemble(run)
    if os.path.isdir(run.args.ic):
        files = rawdata.list_event_files(run.args.ic, run.args.ic_pattern)
        ids = sorted(files)
        reader = ((eid, rawdata.read_grid_file(files[eid], n_pix)) for eid in ids)
    else:
        def lines():
            with open(run.args.ic) as f:
                for i, line in enumerate(f):
                    yield i, np.array(line.split(), dtype=np.float64)
        reader = lines()
    rows, buf = [], []

    def flush():
        ids_, xs = zip(*buf)
        grids, sums = to_grids(np.stack(xs), run.args.ic_kind, cfg)
        nch, pt, mem = ens.predict_grids(grids, sums)
        sd = np.nanstd(mem, axis=0, ddof=1)
        for j, eid in enumerate(ids_):
            rows.append({'event_id': eid, f'S_{cfg["input"]}': sums[cfg['input']][j], 'nch': nch[j], 'pt': pt[j],
                         'nch_fold_spread_rel': sd[j, 0], 'pt_fold_spread_rel': sd[j, 1]})
        buf.clear()
        print(f'  {len(rows)} events predicted', flush=True)

    for eid, x in reader:
        buf.append((eid, x))
        if len(buf) == 256:
            flush()
    if buf:
        flush()
    name = os.path.basename(os.path.normpath(run.args.ic)).split('.')[0]
    path = run.path('predict', f'predictions_{name}.csv')
    pd.DataFrame(rows).to_csv(path, index=False, float_format='%.7g')
    run.log(STEP, f'{len(rows)} events from {run.args.ic} -> {path}')


# ---------------------------------------------------------------------------- B: TRENTo production
def run_trento(p, n, seed, workdir):
    if os.path.exists(os.path.join(workdir, 'events')):
        shutil.rmtree(os.path.join(workdir, 'events'))
    args = p['trento_args'].split()
    cmd = [p['trento_bin']] + args[:-2] + ['--random-seed', str(seed), '--output', 'events'] + args[-2:] + [str(n)]
    out = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f'trento failed ({out.returncode}): {" ".join(cmd)}\n{out.stderr}')
    summary = np.array([l.split() for l in out.stdout.strip().splitlines()], dtype=np.float64)
    assert summary.shape == (n, 8), summary.shape
    # TRENTo zero-pads file names to the width of n-1: order by the integer in the name
    files = sorted(glob.glob(os.path.join(workdir, 'events', '*.dat')), key=lambda f: int(os.path.basename(f).split('.')[0]))
    assert len(files) == n
    E = np.stack([pd.read_csv(f, sep=' ', comment='#', header=None, engine='c').values.ravel() for f in files])
    assert np.allclose(E.sum(1) * 0.01, summary[:, 3], rtol=1e-6), 'grid sum differs from TRENTo multiplicity'
    return summary, E


def workdir(tag):
    base = os.environ.get('SLURM_TMPDIR') or os.path.join('/tmp', os.environ.get('USER', 'user'))
    d = os.path.join(base, f'gouai_trento_{os.environ.get("SLURM_JOB_ID", os.getpid())}_{tag}')
    os.makedirs(d, exist_ok=True)
    return d


def trento_task(run, task, preflight=False):
    cfg, p = run.cfg, run.cfg['production']
    if not p.get('trento_args'):
        sys.exit('set production.trento_args in the config (the TRENTo parameters of the hybrid events)')
    ens = load_ensemble(run)
    if preflight:
        lo, hi, bsz = 0, 20, 20
    else:
        edges = np.linspace(0, p['n_events'], p['n_tasks'] + 1).astype(int)
        lo, hi, bsz = edges[task], edges[task + 1], p['batch']
    nb = int(np.ceil((hi - lo) / bsz))
    out = run.subdir('predict', 'parts', f'task_{task}') if not preflight else None
    wd = workdir(task)
    t0 = time.time()
    for b in range(nb):
        fn = os.path.join(out, f'batch_{b:04d}.npz') if out else None
        if fn and os.path.exists(fn):
            continue
        n = min(bsz, hi - lo - b * bsz)
        seed = p['seed_base'] + 100000 * task + b
        summary, E = run_trento(p, n, seed, wd)
        grids, sums = to_grids(E, 'energy', cfg)
        nch, pt, _ = ens.predict_grids(grids, sums)
        summary[:, 0] += lo + b * bsz
        if preflight:
            for i in range(5):
                print(f'   event {i}: b={summary[i, 1]:.2f} npart={summary[i, 2]:.0f} S={sums[cfg["input"]][i]:.0f} '
                      f'-> N_ch={nch[i]:.2f} [pT]={pt[i]:.4f}')
            ok = sums[cfg['input']] > 0
            assert np.all(np.isfinite(nch[ok])) and np.all((pt[ok] > 0.2) & (pt[ok] < 2.0)), 'implausible predictions'
            run.log(STEP, 'TRENTo preflight OK (20 events)')
            break
        np.savez(fn + '.tmp.npz', summary=summary, S=sums[cfg['input']], nch=nch, pt=pt, seed=seed)
        os.replace(fn + '.tmp.npz', fn)
        if b % 25 == 0 or b == nb - 1:
            print(f'[task {task}] batch {b + 1}/{nb}  {(time.time() - t0) / 60:.1f} min', flush=True)
    shutil.rmtree(wd, ignore_errors=True)
    if not preflight:
        run.log(STEP, f'TRENTo task {task}: events [{lo}, {hi}) done in {(time.time() - t0) / 3600:.2f} h')


def trento_merge(run):
    cfg, p = run.cfg, run.cfg['production']
    files = sorted(glob.glob(os.path.join(run.dir, 'predict', 'parts', 'task_*', 'batch_*.npz')))
    parts = []
    for f in files:
        z = np.load(f)
        df = pd.DataFrame(z['summary'], columns=COLS)
        df[f'S_{cfg["input"]}'], df['nch'], df['pt'] = z['S'], z['nch'], z['pt']
        parts.append(df)
    df = pd.concat(parts).sort_values('event').reset_index(drop=True)
    df['event'] = df['event'].astype(np.int64)
    df['npart'] = df['npart'].astype(np.int64)
    if len(df) != p['n_events'] or not df.event.is_unique:
        sys.exit(f'{len(df)} events found, expected {p["n_events"]}: some batches are missing (re-run those tasks)')
    df['zero_input'] = (df[f'S_{cfg["input"]}'] <= 0).astype(np.int8)
    path = run.path('predict', 'events.csv.gz')
    df.to_csv(path, index=False, float_format='%.7g')
    lo, hi = run.read_json('targets/meta.json')['S_range_outer_train']
    S = df[f'S_{cfg["input"]}']
    write_json(run.path('predict', 'summary.json'), {
        'n_events': len(df), 'file': path, 'columns': list(df.columns), 'model': p['model'],
        'trento_args': p['trento_args'], 'seed_base': p['seed_base'],
        'fraction_S_below_training_min': float(np.mean(S < lo)), 'fraction_S_above_training_max': float(np.mean(S > hi)),
        'n_zero_input_events': int(df.zero_input.sum())})
    run.log(STEP, f'{len(df)} events -> {path}; {100 * np.mean(S < lo):.1f}% below / {100 * np.mean(S > hi):.3f}% '
                  'above the training S range (extrapolated)')


def main():
    run = Run.from_args(__doc__, extra)
    a = run.args
    if a.ic:
        predict_files(run)
    elif a.trento_preflight:
        trento_task(run, 0, preflight=True)
    elif a.trento_merge:
        trento_merge(run)
    else:
        task = a.trento_task if a.trento_task is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID', -1))
        if task < 0:
            sys.exit('choose --ic, --trento-preflight, --trento-task K (or a Slurm array) or --trento-merge')
        trento_task(run, task)


if __name__ == '__main__':
    main()
