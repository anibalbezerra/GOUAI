"""
config.py
---------
Configuration and run-directory handling.

An *experiment* is described by one YAML file (for example configs/examples/PbPb_new_entropy.yaml).
It is merged on top of configs/defaults.yaml, so it only needs the keys that differ.
All outputs of the experiment go to its `run_dir`:

    <run_dir>/
        config.yaml          the fully merged configuration actually used (written by step 01)
        LOG.md               one line per step: what ran, when, and the headline numbers
        data/                step 01: event table and density grids
        splits/              step 02: held-out test set and the cross-validation folds
        targets/             step 03: trend g(S) and normalizations
        smoke/               step 04
        cv/fold_<k>/         step 05: one trained network per fold
        cv/                  step 06: cross-validation report (Gate 2)
        test/                step 07: held-out test report
        compare_v6/          step 08: comparison with the published v6 models
        final/               step 09 (optional)
        predict/             step 10 (optional)

Every script starts with `run = Run.from_args()`; `run.cfg` is the merged dictionary.
"""
import argparse
import copy
import datetime
import json
import os
import sys

import numpy as np
import yaml

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PKG_DIR)
DEFAULTS = os.path.join(REPO_DIR, 'configs', 'defaults.yaml')

SYSTEMS = {
    '2.76_PbPb': {'label': 'Pb-Pb 2.76 TeV', 'tex': r'Pb–Pb $\sqrt{s_{NN}}$ = 2.76 TeV', 'short': 'PbPb'},
    '0.2_AuAu': {'label': 'Au-Au 200 GeV', 'tex': r'Au–Au $\sqrt{s_{NN}}$ = 200 GeV', 'short': 'AuAu'},
}
INPUTS = ('entropy', 'energy')


class ConfigError(Exception):
    """A problem in the experiment configuration, explained in plain words."""


# ----------------------------------------------------------------------------- merging and validation
def deep_merge(base, override):
    """Return base updated recursively with override (dictionaries are merged, anything else replaced)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path):
    """Read defaults.yaml + the experiment file, fill derived values, and check them."""
    with open(DEFAULTS) as f:
        cfg = yaml.safe_load(f)
    with open(path) as f:
        exp = yaml.safe_load(f) or {}
    unknown = set(exp) - set(cfg) - {'description'}
    if unknown:
        raise ConfigError(f'{path}: unknown top-level keys {sorted(unknown)} (typo?). '
                          f'Valid keys: {sorted(cfg)}')
    cfg = deep_merge(cfg, exp)
    cfg['_config_path'] = os.path.abspath(path)
    resolve(cfg)
    return cfg


def resolve(cfg):
    """Fill the values whose default depends on other keys, and validate the essentials."""
    if cfg.get('system') not in SYSTEMS:
        raise ConfigError(f'system must be one of {list(SYSTEMS)}, got {cfg.get("system")!r}')
    if cfg.get('input') not in INPUTS:
        raise ConfigError(f'input must be one of {INPUTS}, got {cfg.get("input")!r}')
    if not cfg.get('run_dir'):
        raise ConfigError('run_dir is not set: every experiment needs its own output directory')
    unedited = [k for k, v in (('run_dir', cfg['run_dir']), ('data.x_path', cfg['data'].get('x_path')),
                               ('data.y_file', cfg['data'].get('y_file')),
                               ('data.reuse_prepared_from', cfg['data'].get('reuse_prepared_from')),
                               ('splits.import_from', cfg['splits'].get('import_from')),
                               ('reference.old_data.x_file', cfg['reference']['old_data'].get('x_file')),
                               ('reference.old_data.y_file', cfg['reference']['old_data'].get('y_file')))
                if v and 'CHANGE' in str(v)]
    if unedited:
        raise ConfigError(f'{", ".join(unedited)} still contain CHANGE: copy the example file and edit the '
                          'lines marked CHANGE, or delete the reference.old_data block if you do not have the v6 data '
                '(docs/04_student_guide.md)')
    cfg['run_dir'] = os.path.abspath(os.path.expanduser(cfg['run_dir']))
    link = linked_top_dir(cfg['run_dir'])
    if link:
        raise ConfigError(f'run_dir {cfg["run_dir"]}: {link}')
    if cfg['data'].get('x_kind') not in INPUTS:
        raise ConfigError(f'data.x_kind must be one of {INPUTS}')
    if cfg['input'] == 'energy' and cfg['data']['x_kind'] == 'entropy':
        raise ConfigError('input: energy needs energy-density files (data.x_kind: energy); '
                          'entropy cannot be converted back to energy')
    if cfg['data'].get('eos_file') is None:
        cfg['data']['eos_file'] = os.path.join(PKG_DIR, 'data', 'hrg_hotqcd_eos_binary.dat')
    if cfg['features'].get('channels') is None:
        cfg['features']['channels'] = [f'{cfg["input"]}_abs', f'{cfg["input"]}_norm']
    if cfg['evaluation'].get('centrality_table') is None:
        cfg['evaluation']['centrality_table'] = os.path.join(
            REPO_DIR, 'reference', 'centrality', f'{cfg["system"]}_{cfg["input"]}.csv')
    if cfg['reference'].get('models_dir') is None:
        cfg['reference']['models_dir'] = os.path.join(REPO_DIR, 'reference', 'v6')
    for section, key in PATH_KEYS:
        holder = cfg['reference']['old_data'] if section == 'reference.old_data' else cfg[section]
        if holder.get(key):
            holder[key] = repo_path(holder[key])
    cfg['label'] = SYSTEMS[cfg['system']]['label']
    return cfg


def linked_top_dir(path):
    """
    A message if the first directory of `path` is a symbolic link on this machine, else ''.
    Example: on some clusters /scratch is a link to /home/scratch on the login node but a different,
    node-local disk on the compute nodes, so jobs would not find the files. The resolved path is safe.
    """
    parts = os.path.abspath(path).split(os.sep)
    top = os.sep + parts[1] if len(parts) > 1 and parts[1] else ''
    if top and os.path.islink(top):
        real = os.path.join(os.path.realpath(top), *parts[2:])
        return (f'{top} is a symbolic link to {os.path.realpath(top)} here, and compute nodes may not have '
                f'the same link: use {real} instead')
    return ''


# paths in the config: relative ones are taken relative to the repository folder
PATH_KEYS = [('data', 'x_path'), ('data', 'y_file'), ('data', 'eos_file'), ('data', 'reuse_prepared_from'),
             ('splits', 'import_from'), ('evaluation', 'centrality_table'), ('reference', 'models_dir'),
             ('reference.old_data', 'x_file'), ('reference.old_data', 'y_file')]


def repo_path(p):
    p = os.path.expanduser(str(p))
    return p if os.path.isabs(p) else os.path.join(REPO_DIR, p)


# ----------------------------------------------------------------------------- the run directory
class Run:
    """
    Paths, logging and small I/O helpers for one experiment.

        run = Run.from_args()                     # parses --config (and any extra arguments you add)
        run.path('data', 'event_table.csv')       # <run_dir>/data/event_table.csv (parent directory created)
        run.require('splits/splits.json', '02_make_splits.py')
        run.log('05_train_fold', 'fold 0: best epoch 230')
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.dir = cfg['run_dir']

    @classmethod
    def from_args(cls, description=None, extra=None):
        """
        Parse the command line. `extra(parser)` may add script-specific arguments;
        they are returned in run.args.
        """
        ap = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument('--config', required=True, help='experiment YAML file')
        if extra:
            extra(ap)
        args = ap.parse_args()
        try:
            run = cls(load_config(args.config))
        except ConfigError as e:
            sys.exit(f'CONFIG ERROR: {e}')
        run.args = args
        os.makedirs(run.dir, exist_ok=True)
        return run

    # paths -----------------------------------------------------------------
    def path(self, *parts):
        p = os.path.join(self.dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def subdir(self, *parts):
        p = os.path.join(self.dir, *parts)
        os.makedirs(p, exist_ok=True)
        return p

    def exists(self, *parts):
        return os.path.exists(os.path.join(self.dir, *parts))

    def require(self, rel, step):
        """Stop with a clear message if an earlier step has not produced `rel` yet."""
        if not self.exists(rel):
            sys.exit(f'\nMISSING: {os.path.join(self.dir, rel)}\n'
                     f'Run step {step} first (see docs/04_student_guide.md).\n')

    # logging ---------------------------------------------------------------
    def log(self, step, msg):
        """Print a message and append it, time-stamped, to <run_dir>/LOG.md."""
        stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        job = os.environ.get('SLURM_ARRAY_JOB_ID') or os.environ.get('SLURM_JOB_ID', 'interactive')
        task = os.environ.get('SLURM_ARRAY_TASK_ID')
        jid = f'{job}_{task}' if task is not None else job
        line = f'- `{stamp}` **{step}** (job {jid}): {msg}'
        print(line, flush=True)
        with open(os.path.join(self.dir, 'LOG.md'), 'a') as f:
            f.write(line + '\n')

    def save_config(self):
        """Write the merged configuration next to the results (provenance)."""
        cfg = {k: v for k, v in self.cfg.items() if not k.startswith('_')}
        with open(self.path('config.yaml'), 'w') as f:
            f.write(f'# merged configuration, from {self.cfg["_config_path"]}\n')
            yaml.safe_dump(cfg, f, sort_keys=False)

    # json ------------------------------------------------------------------
    def write_json(self, rel, obj):
        write_json(self.path(rel), obj)

    def read_json(self, rel):
        return read_json(os.path.join(self.dir, rel))


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=_json_default)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    raise TypeError(type(o))
