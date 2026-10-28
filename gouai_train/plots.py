"""
plots.py
--------
All figures written by the step scripts. Each function takes plain arrays/DataFrames and a path
(without extension) and writes <path>.png and <path>.pdf in the style of style.py.

    data_overview        step 01  events per centrality class (new sample vs v6 sample), N_ch(S), [pT](S)
    trend_fit            step 03  the trend g(S) and the residual d = log y - log g(S) of the training events
    learning_curves      step 06  training/validation loss of every fold, warm-up shaded, kept epoch marked
    parity               step 06/07  predicted vs true N_ch and [pT]
    residual_recovery    step 06/07  predicted vs true residual beyond g(S): the evidence that the network
                                     knows more than S
    error_vs_centrality  step 07/08  mean relative error per centrality class, one line per model
    fold_spread          step 07  disagreement between the fold networks per centrality class
"""
import numpy as np

from . import style
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

from .style import plt, NEW, V6, TREND, FOLD_SHADES

style.apply()

LABEL = {'nch': r'$N_{\rm ch}$', 'pt': r'$[p_T]$ (GeV)'}
SHORT = {'nch': r'$N_{\rm ch}$', 'pt': r'$[p_T]$'}


def data_overview(cent_new, S, nch, pt, edges, path, cent_ref=None, title=''):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    w = np.diff(edges)
    h_new = np.histogram(cent_new, edges)[0]
    x = np.arange(len(w))
    ax[0].bar(x - 0.2, h_new, 0.4, color=NEW, label=f'this sample ({len(cent_new)})')
    if cent_ref is not None:
        h_ref = np.histogram(cent_ref, edges)[0]
        ax[0].bar(x + 0.2, h_ref, 0.4, color=V6, label=f'v6 sample ({len(cent_ref)})')
    ax[0].set_xticks(x, [f'{a:g}-{b:g}' for a, b in zip(edges[:-1], edges[1:])], rotation=60, fontsize=8)
    ax[0].xaxis.set_minor_locator(NullLocator())
    ax[0].set_xlabel('minimum-bias centrality class (%)'); ax[0].set_ylabel('events')
    ax[0].set_ylim(0, 1.3 * max(np.max(h_new), np.max(h_ref) if cent_ref is not None else 0))
    ax[0].legend(loc='upper right'); ax[0].set_title('(a)')
    ax[1].loglog(S, nch, '.', ms=3, color=NEW); ax[1].set_xlabel('S'); ax[1].set_ylabel(LABEL['nch']); ax[1].set_title('(b)')
    ax[2].semilogx(S, pt, '.', ms=3, color=NEW); ax[2].set_xlabel('S'); ax[2].set_ylabel(LABEL['pt']); ax[2].set_title('(c)')
    if title:
        fig.suptitle(title, fontweight='bold')
    style.save(fig, path)


def trend_fit(spec, S, y, path, title=''):
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    grid = np.geomspace(S.min(), S.max(), 200)
    for j, name in enumerate(('nch', 'pt')):
        ax[0, j].loglog(S, y[name], '.', ms=3, color=NEW, alpha=0.6)
        ax[0, j].loglog(grid, spec.trends[name](grid) * np.exp(spec.mu[name]), '-', color=V6, lw=1.5)
        ax[0, j].set_ylabel(LABEL[name])
        d = spec.residual(S, y[name], name)
        ax[1, j].semilogx(S, d, '.', ms=3, color=NEW, alpha=0.6)
        ax[1, j].axhline(spec.mu[name], color=V6, lw=1.2)
        ax[1, j].set_ylabel(r'$d = \log y - \log g(S)$'); ax[1, j].set_xlabel('S')
        ax[1, j].set_title(f'σ_d = {spec.sigma[name]:.4f}')
    if title:
        fig.suptitle(title, fontweight='bold')
    style.save(fig, path)


def learning_curves(histories, warmup, path, best_epochs=None, title=''):
    """histories: {label: DataFrame with loss/val_loss}; best_epochs: {label: epoch}."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axvspan(0, warmup, color='#e1e0d9', alpha=0.6, lw=0, label='warm-up')
    for i, (lab, h) in enumerate(histories.items()):
        c = FOLD_SHADES[i % len(FOLD_SHADES)]
        ep = np.arange(1, len(h) + 1)
        ax.semilogy(ep, h['loss'], '--', color=c, lw=1)
        if 'val_loss' in h:
            ax.semilogy(ep, h['val_loss'], '-', color=c, lw=1.4, label=lab)
            if best_epochs and lab in best_epochs:
                b = best_epochs[lab]
                ax.semilogy([b], [h['val_loss'].iloc[b - 1]], 'o', color=c, ms=6)
    ax.set_xlabel('epoch'); ax.set_ylabel('loss (dashed: train, solid: validation)')
    ax.legend(loc='upper right', fontsize=8)
    if title:
        ax.set_title(title)
    style.save(fig, path)


def parity(sets, path, title=''):
    """sets: {label: (y_true dict, y_pred dict, color, marker)}"""
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.3))
    for lab, (yt, yp, c, mk) in sets.items():
        for j, name in enumerate(('nch', 'pt')):
            ax[j].plot(yt[name], yp[name], ls='none', marker=mk, ms=4, mfc=c, mec='white', mew=0.4,
                       alpha=0.85, label=lab)
    for j, name in enumerate(('nch', 'pt')):
        lo = min(l.get_xdata().min() for l in ax[j].get_lines())
        hi = max(l.get_xdata().max() for l in ax[j].get_lines())
        ax[j].plot([lo, hi], [lo, hi], '-', color='black', lw=1, zorder=0)
        ax[j].set_xlabel(f'{LABEL[name]} hybrid'); ax[j].set_ylabel(f'{LABEL[name]} network')
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].legend(loc='lower right')
    if title:
        fig.suptitle(title, fontweight='bold')
    style.save(fig, path)


def residual_recovery(sets, path, title=''):
    """sets: {label: (d_true dict, d_pred dict, color)} residuals beyond g_ref."""
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.3))
    for lab, (dt, dp, c) in sets.items():
        for j, name in enumerate(('nch', 'pt')):
            ax[j].plot(dt[name], dp[name], '.', ms=4, color=c, alpha=0.7, label=lab)
    for j, name in enumerate(('nch', 'pt')):
        lim = max(np.percentile(np.abs(np.r_[tuple(np.r_[s[0][name], s[1][name]] for s in sets.values())]), 99.5) * 1.15, 1e-3)
        ax[j].plot([-lim, lim], [-lim, lim], '-', color='black', lw=1)
        ax[j].set_xlim(-lim, lim); ax[j].set_ylim(-lim, lim)
        ax[j].set_xlabel(f'true residual of {SHORT[name]}')
        ax[j].set_ylabel('predicted residual')
    ax[0].legend(loc='upper left')
    if title:
        fig.suptitle(title, fontweight='bold')
    style.save(fig, path)


def error_vs_centrality(models, path, title='', cis=None):
    """
    models: {label: (metrics from metrics.evaluate, color)}
    cis: optional {label: {'nch': [(lo, hi) per class], 'pt': [...]}} error bars (relative error).
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for lab, (m, c) in models.items():
        for j, name in enumerate(('nch', 'pt')):
            cl = [k for k in m[name]['classes'] if k['n'] > 0]
            xc = np.array([(k['lo'] + k['hi']) / 2 for k in cl])
            v = np.array([100 * k['mean_rel_err'] for k in cl])
            xerr = np.array([(k['hi'] - k['lo']) / 2 for k in cl])
            yerr = None
            if cis and lab in cis:
                b = np.array([cis[lab][name][i] for i, k in enumerate(m[name]['classes']) if k['n'] > 0], float)
                yerr = np.abs(np.vstack([v - 100 * b[:, 0], 100 * b[:, 1] - v]))
            ax[j].errorbar(xc, v, xerr=xerr, yerr=yerr, fmt='o', color=c, ms=5, lw=1.4, capsize=2, label=lab)
    for j, name in enumerate(('nch', 'pt')):
        ax[j].set_xlabel('minimum-bias centrality (%)')
        ax[j].set_ylabel(f'mean |relative error| of {SHORT[name]} (%)')
        ax[j].set_xlim(0, 100); ax[j].set_yscale('log')    # 0.5 % ... 50 %: a log axis shows every class
        ax[j].yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]))
        ax[j].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax[j].yaxis.set_minor_formatter(NullFormatter())
    ax[0].legend(loc='upper left')
    if title:
        fig.suptitle(title, fontweight='bold')
    style.save(fig, path)


def fold_spread(cent, members, edges, path, title=''):
    """members: (n_folds, n_events, 2) log predictions; plots the per-class median spread in %."""
    sd = np.std(members, axis=0, ddof=1)               # (n_events, 2) std of log y = relative spread
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    k = np.digitize(cent, edges[1:-1])
    for j, (name, c) in enumerate((('nch', NEW), ('pt', V6))):
        xc, v = [], []
        for b in range(len(edges) - 1):
            s = k == b
            if s.any():
                xc.append((edges[b] + edges[b + 1]) / 2); v.append(100 * np.median(sd[s, j]))
        ax.plot(xc, v, 'o-', color=c, ms=5, lw=1.4, label=SHORT[name])
    ax.set_xlabel('minimum-bias centrality (%)'); ax.set_ylabel('fold-to-fold spread (%)')
    ax.set_xlim(0, 100); ax.set_ylim(bottom=0); ax.legend(loc='upper left')
    if title:
        ax.set_title(title)
    style.save(fig, path)
