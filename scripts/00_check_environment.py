#!/usr/bin/env python3
"""
STEP 00 - check the software and your configuration            (login node or GPU job, seconds)

    python scripts/00_check_environment.py --config configs/my_experiment.yaml
    python scripts/00_check_environment.py --config ... --gpu        (inside a GPU job: also checks the GPU)

Prints one line per check: OK, WARN (works, but read the message) or FAIL (fix before going on).
"""
import importlib
import os
import sys

import _path  # noqa: F401

RESULTS = []


def report(status, what, detail=''):
    RESULTS.append(status)
    print(f'[{status:4s}] {what}' + (f': {detail}' if detail else ''), flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--gpu', action='store_true', help='also check that TensorFlow sees a GPU')
    a = ap.parse_args()

    from gouai_train.config import linked_top_dir
    link = linked_top_dir(os.environ.get('PWD', os.getcwd()))
    if link:
        report('FAIL', 'working directory', link + ' (cd there before running or submitting anything)')
    v = sys.version_info
    report('OK' if v >= (3, 10) else 'FAIL', 'python version', f'{v.major}.{v.minor}.{v.micro} ({sys.executable})')
    for mod in ('numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'yaml', 'h5py'):
        try:
            m = importlib.import_module(mod)
            report('OK', f'package {mod}', getattr(m, '__version__', ''))
        except ImportError:
            report('FAIL', f'package {mod}', 'missing: pip install -r requirements.txt')

    from gouai_train.config import load_config, ConfigError
    try:
        cfg = load_config(a.config)
        report('OK', 'configuration', a.config)
    except ConfigError as e:
        report('FAIL', 'configuration', str(e))
        return finish()
    d = cfg['data']
    if d.get('reuse_prepared_from'):
        ok = os.path.exists(os.path.join(d['reuse_prepared_from'], 'data', 'event_table.csv'))
        report('OK' if ok else 'FAIL', 'data.reuse_prepared_from', d['reuse_prepared_from'])
    else:
        for key in ('x_path', 'y_file'):
            p = d.get(key)
            report('OK' if p and os.path.exists(p) else 'FAIL', f'data.{key}', str(p))
            if p and linked_top_dir(p):
                report('WARN', f'data.{key}', linked_top_dir(p))
    for key, p in (('centrality table', cfg['evaluation']['centrality_table']), ('EoS table', d['eos_file'])):
        report('OK' if os.path.exists(p) else 'FAIL', key, p)
    try:
        os.makedirs(cfg['run_dir'], exist_ok=True)
        t = os.path.join(cfg['run_dir'], '.write_test')
        open(t, 'w').close(); os.remove(t)
        report('OK', 'run_dir writable', cfg['run_dir'])
    except OSError as e:
        report('FAIL', 'run_dir writable', str(e))
    old = cfg['reference']['old_data']
    if old.get('y_file'):
        report('OK' if os.path.exists(old['y_file']) else 'FAIL', 'reference.old_data.y_file', old['y_file'])
    else:
        report('WARN', 'reference.old_data', 'not set: no protection of the v6 test events, no comparison B in step 08')
    w = os.path.join(cfg['reference']['models_dir'], cfg['system'], f'{cfg["input"]}_pt', 'fold_0', 'model.weights.h5')
    report('OK' if os.path.exists(w) else 'WARN', 'v6 weights (step 08)',
           w if os.path.exists(w) else 'not fetched yet: python scripts/fetch_v6_weights.py (needed only for step 08)')

    try:
        import tensorflow as tf
        report('OK', 'tensorflow', tf.__version__)
        if a.gpu:
            from gouai_train import gpu_env  # noqa: F401
            gpus = tf.config.list_physical_devices('GPU')
            report('OK' if gpus else 'FAIL', 'GPU visible to TensorFlow', str(gpus) or 'none')
    except ImportError:
        report('FAIL', 'tensorflow', 'missing: pip install -r requirements.txt')
    return finish()


def finish():
    n_fail = RESULTS.count('FAIL')
    print(f'\n{"ALL CHECKS PASSED" if n_fail == 0 else f"{n_fail} check(s) FAILED"} '
          f'({RESULTS.count("WARN")} warning(s))')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
