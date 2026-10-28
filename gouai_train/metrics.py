"""
metrics.py
----------
One set of accuracy measures for every model (trend alone, new networks, v6 networks).

Physical accuracy (what the paper quotes)
    mae            mean |y_pred - y_true|           ([pT] in MeV)
    mean_rel_err   mean |y_pred / y_true - 1|
    bias           mean (y_pred - y_true)

Residual accuracy (does the network know more than S?)
All models are measured against the same reference trend g_ref(S), fitted on the training events:
    d_true = log y_true - log g_ref(S),    d_pred = log y_pred - log g_ref(S)
    r          correlation of d_true and d_pred
    beta_cal   cov(d_true, d_pred) / var(d_pred)   calibration slope: the fraction of the predicted
               residual that is real (1 = calibrated, 0 = pure noise, > 1 = too timid)
    err_ratio  rmse(d_pred - d_true) / std(d_true): < 1 means the network beats the trend g(S) alone
               (0.4 means it removes 1 - 0.4^2 = 84 % of the variance S cannot explain)

Subsets
    all            every event
    nch_ge_floor   N_ch >= targets.nch_floor (10): the residual of N_ch in the most peripheral events
                   is dominated by noise, so the N_ch residual criteria use this subset
    classes        per minimum-bias centrality class (evaluation.centrality_edges)
    class_avg      the per-class mean relative errors averaged over evaluation.summary_range (0-80 %),
                   each class weighted by its WIDTH. This is what a minimum-bias sample would see, and,
                   unlike the plain average over test events, it does NOT depend on how many test events
                   fall in each class. Use it to compare samples with different centrality composition
                   (e.g. the new sample, rich in ultra-central events, with the v6 sample).
"""
import numpy as np
import pandas as pd

from .centrality import class_index, class_labels
from .targets import TARGETS

TINY_STD = 1e-9   # a constant prediction (the trend alone) has std ~1e-16 from round-off


def _corr(a, b):
    if len(a) < 3 or np.std(a) < TINY_STD or np.std(b) < TINY_STD:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def residual_block(d_true, d_pred):
    if len(d_true) < 3:
        return {'n': int(len(d_true))}
    c = np.cov(d_true, d_pred, ddof=1)
    var_t, var_p, cov = c[0, 0], c[1, 1], c[0, 1]
    err = d_pred - d_true
    return {'n': int(len(d_true)),
            'r': _corr(d_true, d_pred),
            'beta': float(cov / var_t) if var_t > TINY_STD ** 2 else float('nan'),
            'beta_cal': float(cov / var_p) if var_p > TINY_STD ** 2 else float('nan'),
            'std_d_true': float(np.sqrt(var_t)),
            'rmse_d': float(np.sqrt(np.mean(err ** 2))),
            'err_ratio': float(np.sqrt(np.mean(err ** 2)) / np.sqrt(var_t)) if var_t > 0 else float('nan')}


def physical_block(y_true, y_pred, scale=1.0):
    if len(y_true) == 0:
        return {'n': 0}
    err = np.abs(y_pred - y_true)
    return {'n': int(len(y_true)),
            'mae': float(err.mean() * scale),
            'mean_rel_err': float(np.mean(err / y_true)),
            'bias': float(np.mean(y_pred - y_true) * scale)}


def evaluate(S, y_true, y_pred, gref, cent, cfg):
    """
    S, cent : (n,) total S and minimum-bias centrality (%) of each event
    y_true, y_pred : {'nch': (n,), 'pt': (n,)} physical units, [pT] in GeV
    gref    : targets.ResidualSpec of the reference trend (only its trends are used)
    Returns {'nch': {...}, 'pt': {...}} with the subsets described above.
    """
    S = np.asarray(S, np.float64)
    ev = cfg['evaluation']
    edges = ev['centrality_edges']
    labels = class_labels(edges)
    k = class_index(cent, edges)
    lo, hi = ev['summary_range']
    out = {}
    for name in TARGETS:
        yt = np.asarray(y_true[name], np.float64)
        yp = np.asarray(y_pred[name], np.float64)
        floor = cfg['targets']['pred_floor_frac'] * yt.max()
        n_clip = int(np.sum(yp <= floor))
        log_g = gref.trends[name].log_g(S)
        d_t = np.log(yt) - log_g
        d_p = np.log(np.maximum(yp, floor)) - log_g
        scale = 1e3 if name == 'pt' else 1.0                 # [pT] errors in MeV
        m = {'all': {**physical_block(yt, yp, scale), **residual_block(d_t, d_p)},
             'n_nonpositive_clipped': n_clip}
        sel = np.asarray(y_true['nch']) >= cfg['targets']['nch_floor']
        if sel.sum() > 2:
            m['nch_ge_floor'] = {**physical_block(yt[sel], yp[sel], scale), **residual_block(d_t[sel], d_p[sel])}
        m['classes'] = []
        for j, lab in enumerate(labels):
            s = k == j
            m['classes'].append({'class': lab, 'lo': edges[j], 'hi': edges[j + 1],
                                 **physical_block(yt[s], yp[s], scale), **residual_block(d_t[s], d_p[s])})
        m['class_avg'] = class_average(m['classes'], lo, hi)
        out[name] = m
    return out


def class_average(classes, lo, hi):
    """Width-weighted mean of the per-class mean relative errors over [lo, hi] %."""
    inside = [c for c in classes if c['lo'] >= lo and c['hi'] <= hi]
    filled = [c for c in inside if c['n'] > 0]
    if not filled:
        return {'range': f'{lo:g}-{hi:g}%', 'mean_rel_err': float('nan'), 'covered_fraction': 0.0}
    w = np.array([c['hi'] - c['lo'] for c in filled], float)
    v = np.array([c['mean_rel_err'] for c in filled])
    return {'range': f'{lo:g}-{hi:g}%', 'mean_rel_err': float(np.sum(w * v) / w.sum()),
            'covered_fraction': float(w.sum() / (hi - lo)),
            'empty_classes': [c['class'] for c in inside if c['n'] == 0]}


def paired_bootstrap(err_a, err_b, n_boot, seed):
    """
    Model A vs model B on the SAME events. err_a, err_b: per-event |relative error|.
    Returns mean difference (A - B) and its 95 % bootstrap interval (events resampled with replacement).
    If the interval excludes 0, the difference is unlikely to be a fluctuation of the test sample.
    """
    err_a, err_b = np.asarray(err_a), np.asarray(err_b)
    n = len(err_a)
    if n < 3:
        return {'n': n, 'diff': float('nan'), 'lo': float('nan'), 'hi': float('nan')}
    rng = np.random.default_rng(seed)
    diff = err_a - err_b
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return {'n': n, 'diff': float(diff.mean()), 'lo': float(np.percentile(boots, 2.5)),
            'hi': float(np.percentile(boots, 97.5))}


# ----------------------------------------------------------------------------- tables
def flat_row(model_name, metrics, subset='all'):
    """One table row with the headline numbers of an evaluate() result."""
    row = {'model': model_name}
    for name in TARGETS:
        m = metrics[name].get(subset)
        if m is None:
            continue
        for key in ('n', 'mae', 'mean_rel_err', 'r', 'beta_cal', 'err_ratio'):
            row[f'{name}_{key}'] = m.get(key, float('nan'))
        row[f'{name}_class_avg'] = metrics[name]['class_avg']['mean_rel_err']
    return row


HEADERS = {'nch_n': 'n', 'nch_mae': 'Nch MAE', 'nch_mean_rel_err': 'Nch rel', 'nch_class_avg': 'Nch rel (class avg)',
           'nch_r': 'r_N', 'nch_beta_cal': 'β_N', 'nch_err_ratio': 'err/σ_N', 'pt_mae': 'pT MAE [MeV]',
           'pt_mean_rel_err': 'pT rel', 'pt_class_avg': 'pT rel (class avg)', 'pt_r': 'r_p',
           'pt_beta_cal': 'β_p', 'pt_err_ratio': 'err/σ_p'}


def markdown_table(rows, cols=None):
    df = pd.DataFrame(rows)
    order = ['model'] + list(HEADERS)
    df = df[[c for c in (cols or order) if c in df.columns]].rename(columns=HEADERS)
    lines = ['| ' + ' | '.join(df.columns) + ' |', '|' + '---|' * len(df.columns)]
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(_fmt(v, c) for c, v in zip(df.columns, r.values)) + ' |')
    return '\n'.join(lines)


def _fmt(v, col=''):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return '–'
        if 'rel' in col:
            return f'{100 * v:.2f}%'
        return f'{v:.3f}' if abs(v) < 100 else f'{v:.1f}'
    return str(v)


def class_table(per_model, target):
    """Markdown table: mean relative error per centrality class (rows) for several models (columns)."""
    names = list(per_model)
    first = per_model[names[0]][target]['classes']
    lines = ['| class | n | ' + ' | '.join(names) + ' |', '|---|---|' + '---|' * len(names)]
    for j, c in enumerate(first):
        vals = [per_model[m][target]['classes'][j] for m in names]
        cells = [f'{100 * v["mean_rel_err"]:.2f}%' if v['n'] > 0 else '–' for v in vals]
        lines.append(f'| {c["class"]} | {c["n"]} | ' + ' | '.join(cells) + ' |')
    avg = [per_model[m][target]['class_avg'] for m in names]
    lines.append(f'| **avg {avg[0]["range"]}** | | ' +
                 ' | '.join(f'**{100 * a["mean_rel_err"]:.2f}%**' for a in avg) + ' |')
    return '\n'.join(lines)
