#!/usr/bin/env python3
"""
STEP 08 - compare the new networks with the published v6 networks     (1 GPU recommended, ~10-30 min)

    python scripts/fetch_v6_weights.py --from-dir <folder with the v6 weight files>    (once)
    python scripts/08_compare_v6.py --config configs/my_experiment.yaml

The question of the project: does the new, larger sample (with many more ultra-central events) give
better networks than the ~450 events of v6? Comparing the number in test/report.md with the number in
the paper is NOT a fair answer: the two test sets differ in size and in centrality composition, and the
error depends strongly on centrality. This step compares the two ensembles on the SAME events:

  A. NEW test set: the held-out events of this run, predicted by the new ensemble (from step 05) and by
     the v6 ensemble. Events that v6 used for TRAINING are excluded (v6 would have seen them).
  B. v6 test set: the 169 (Pb-Pb) / 135 (Au-Au) events v6 was validated on, predicted by both
     ensembles. Needs reference.old_data (x_file, y_file). Events of it that the new networks trained
     on are excluded (none if splits.protect_reference_test was on).
     The v6 predictions are also checked against the published ones (reference/v6/.../heldout_predictions.csv)
     to prove that the v6 networks are loaded and fed correctly.

For each, per centrality class and class-averaged over 0-80 %: the mean relative errors of both
models and their difference (new - v6) with a 95 % paired-bootstrap interval. A negative difference whose
interval excludes 0 means the new networks are significantly better on these events.

Writes <run_dir>/compare_v6/report.md, comparison.json, predictions_A.csv, predictions_B.csv and figures.
"""
import os

import numpy as np
import pandas as pd

import _path  # noqa: F401
from gouai_train.config import Run, read_json
from gouai_train.store import EventStore
from gouai_train.targets import ResidualSpec
from gouai_train.centrality import CentralityTable, class_index, class_labels
from gouai_train import metrics as MT
from gouai_train import plots, eos, splits as SP
from gouai_train.style import NEW, V6

STEP = '08_compare_v6'
NOISE = 1e-4      # differences below 0.01 % are numerical noise (GPU vs CPU arithmetic), never "significant"


def compare(cfg, S, cent, yt, preds, gref):
    """Metrics of both models and paired bootstrap of their per-event |relative error| difference."""
    ev = cfg['evaluation']
    mets = {k: MT.evaluate(S, yt, p, gref, cent, cfg) for k, p in preds.items()}
    edges = ev['centrality_edges']
    k = class_index(cent, edges)
    lo, hi = ev['summary_range']
    w = np.diff(edges)
    boot, cis = {}, {m: {} for m in preds}
    for name in ('nch', 'pt'):
        err = {m: np.abs(p[name] / yt[name] - 1) for m, p in preds.items()}
        rows = []
        for j, lab in enumerate(class_labels(edges)):
            s = k == j
            b = MT.paired_bootstrap(err['new'][s], err['v6'][s], ev['n_bootstrap'], ev['bootstrap_seed'] + j)
            rows.append({'class': lab, **b})
        # class-averaged difference: resample within each class, combine with the class widths
        rng = np.random.default_rng(ev['bootstrap_seed'])
        sel = [j for j in range(len(w)) if edges[j] >= lo and edges[j + 1] <= hi and np.sum(k == j) > 0]
        diffs = {j: err['new'][k == j] - err['v6'][k == j] for j in sel}
        ww = np.array([w[j] for j in sel], float)
        point = float(np.sum(ww * [diffs[j].mean() for j in sel]) / ww.sum()) if sel else float('nan')
        bs = [np.sum(ww * [diffs[j][rng.integers(0, len(diffs[j]), len(diffs[j]))].mean() for j in sel]) / ww.sum()
              for _ in range(ev['n_bootstrap'])] if sel else [np.nan]
        boot[name] = {'classes': rows, 'class_avg': {'range': f'{lo:g}-{hi:g}%', 'diff': point,
                                                     'lo': float(np.percentile(bs, 2.5)), 'hi': float(np.percentile(bs, 97.5))}}
        for m in preds:          # bootstrap interval of each model's own per-class error (for the figure)
            cis[m][name] = []
            for j in range(len(w)):
                e = err[m][k == j]
                if len(e) < 2:
                    cis[m][name].append((np.nan, np.nan))
                    continue
                r = np.random.default_rng(j)
                bb = [e[r.integers(0, len(e), len(e))].mean() for _ in range(500)]
                cis[m][name].append((np.percentile(bb, 16), np.percentile(bb, 84)))
    return mets, boot, cis


def section(title, n, notes, mets, boot):
    out = [f'## {title}\n', f'{n} events. ' + ' '.join(notes) + '\n']
    for name, lab in (('nch', 'N_ch'), ('pt', '[pT]')):
        out += [f'### {lab}: mean relative error per centrality class\n',
                '| class | n | new | v6 | new − v6 [95% CI] |', '|---|---|---|---|---|']
        for c_new, c_v6, b in zip(mets['new'][name]['classes'], mets['v6'][name]['classes'], boot[name]['classes']):
            if c_new['n'] == 0:
                out.append(f'| {c_new["class"]} | 0 | – | – | – |')
                continue
            cell = f'{100 * b["diff"]:+.2f}% [{100 * b["lo"]:+.2f}, {100 * b["hi"]:+.2f}]'
            if (b['hi'] < 0 or b['lo'] > 0) and abs(b['diff']) > NOISE:   # interval excludes 0: significant
                cell = f'**{cell}**'
            out.append(f'| {c_new["class"]} | {c_new["n"]} | {100 * c_new["mean_rel_err"]:.2f}% | '
                       f'{100 * c_v6["mean_rel_err"]:.2f}% | {cell} |')
        a, an, av = boot[name]['class_avg'], mets['new'][name]['class_avg'], mets['v6'][name]['class_avg']
        out.append(f'| **avg {a["range"]}** | | **{100 * an["mean_rel_err"]:.2f}%** | **{100 * av["mean_rel_err"]:.2f}%** | '
                   f'**{100 * a["diff"]:+.2f}% [{100 * a["lo"]:+.2f}, {100 * a["hi"]:+.2f}]** |')
        if an.get('empty_classes'):
            out.append(f'\n(no events in {", ".join(an["empty_classes"])}: the average covers '
                       f'{100 * an["covered_fraction"]:.0f}% of the range)')
        out.append('')
    out += ['### All events of this set\n',
            MT.markdown_table([MT.flat_row(k, v) for k, v in mets.items()]), '']
    return out


def read_rows(path, rows, n_pix):
    """The given (sorted) line numbers of a one-event-per-line grid file."""
    want, out = set(int(r) for r in rows), {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i in want:
                a = np.array(line.split(), dtype=np.float64)
                assert a.size == n_pix, f'{path} line {i + 1}: {a.size} numbers'
                out[i] = a
                if len(out) == len(want):
                    break
    return np.stack([out[int(r)] for r in rows])


def main():
    run = Run.from_args(__doc__)
    cfg = run.cfg
    run.require('test/metrics.json', '07_test_report.py')
    from gouai_train.ensemble import Ensemble          # imports TensorFlow
    system, inp = cfg['system'], cfg['input']
    gref = ResidualSpec.from_dict(run.read_json('targets/gref.json'))
    cent_t = CentralityTable(cfg['evaluation']['centrality_table'])
    v6 = Ensemble.from_v6(cfg['reference']['models_dir'], system, inp)
    out = run.subdir('compare_v6')
    result, rep = {}, [f'# New networks vs published v6: {cfg["label"]}, input {inp}\n',
                       'Both ensembles predict the SAME events. Differences are new − v6 in mean |relative error| '
                       '(negative = new is better); bold = the 95% paired-bootstrap interval excludes 0 '
                       '(differences below 0.01% are numerical noise and never bold).\n']

    # ------------------------------------------------------------------ A: the new held-out events
    store = EventStore(run)
    tab = store.table
    te = np.array(run.read_json('splits/splits.json')['outer_test'])
    notes = []
    if 'v6_part' in tab:
        seen = tab.v6_part.fillna('').values[te] == 'train'
        if seen.any():
            notes.append(f'{int(seen.sum())} test events were v6 TRAINING events and are excluded.')
        te = te[~seen]
    else:
        notes.append('Overlap with the v6 training sample unknown (reference.old_data.y_file not set): '
                     'if the new sample contains v6 training events, v6 is favoured here.')
    ens_new = pd.read_csv(run.path('test', 'predictions.csv')).set_index('idx').loc[te]
    n6, p6, m6 = v6.predict_store(store, te)
    S = store.S(te)
    yt = store.y(te)
    preds = {'new': {'nch': ens_new.nch_pred.values, 'pt': ens_new.pt_pred_GeV.values}, 'v6': {'nch': n6, 'pt': p6}}
    mets, boot, cis = compare(cfg, S, cent_t(S), yt, preds, gref)
    result['A_new_test'] = {'metrics': mets, 'bootstrap': boot, 'n': len(te)}
    pd.DataFrame({'idx': te, 'centrality': cent_t(S), 'nch_true': yt['nch'], 'nch_new': preds['new']['nch'], 'nch_v6': n6,
                  'pt_true': yt['pt'], 'pt_new': preds['new']['pt'], 'pt_v6': p6}).to_csv(
        os.path.join(out, 'predictions_A.csv'), index=False, float_format='%.7g')
    rep += section('A. Held-out events of the NEW sample', len(te), notes, mets, boot)
    plots.error_vs_centrality({'new': (mets['new'], NEW), 'v6': (mets['v6'], V6)}, os.path.join(out, 'A_new_test'),
                              f'{cfg["label"]}: new test set', cis=cis)
    rep.append('![A](A_new_test.png)\n')

    # ------------------------------------------------------------------ B: the v6 held-out events
    old = cfg['reference']['old_data']
    if old.get('x_file') and old.get('y_file'):
        ycfg = {'data': {'y_file': old['y_file'], 'y_columns': {'id': 0, 'nch': 1, 'pt': 2}, 'extra_y_columns': {}}}
        from gouai_train import rawdata
        ref = rawdata.read_labels(ycfg)
        vsplit = read_json(os.path.join(cfg['reference']['models_dir'], system, 'splits.json'))
        vte = np.array(vsplit['outer_test'])
        notes = []
        m = SP.match_events(ref.nch.values[vte], ref.pt.values[vte], tab.nch.values, tab.pt.values)
        in_new_train = np.isin(m, run.read_json('splits/splits.json')['outer_train']) & (m >= 0)
        if in_new_train.any():
            notes.append(f'{int(in_new_train.sum())} of these events are in the NEW training set and are excluded.')
        vte = vte[~in_new_train]
        x = read_rows(old['x_file'], vte, cfg['data']['grid_n'] ** 2)
        grids = {'entropy': eos.entropy_density(x, cfg['data']['eos_file']), 'energy': x} \
            if old['x_kind'] == 'energy' else {'entropy': x}
        if inp not in grids:
            raise SystemExit('input energy needs reference.old_data.x_kind: energy (the old energy-density file)')
        grids = {d: g.reshape(-1, cfg['data']['grid_n'], cfg['data']['grid_n']) for d, g in grids.items()}
        sums = {d: g.reshape(len(g), -1).sum(1) for d, g in grids.items()}
        n6, p6, _ = v6.predict_grids(grids, sums)
        new = Ensemble.from_run(run)
        nn, pn, _ = new.predict_grids(grids, sums)
        pub = pd.read_csv(os.path.join(cfg['reference']['models_dir'], system, f'{inp}_pt', 'heldout_predictions.csv')
                          ).set_index('idx_raw').loc[vte]
        dev = float(np.max(np.abs(n6 / pub.nch_nn.values - 1)))
        notes.append(f'v6 predictions reproduce the published ones to {dev:.1e} (relative, max over events).')
        if dev > 1e-3:
            notes.append('**WARNING: v6 predictions differ from the published ones: check reference.old_data and the weights.**')
        S = sums[inp]
        yt = {'nch': ref.nch.values[vte], 'pt': ref.pt.values[vte]}
        preds = {'new': {'nch': nn, 'pt': pn}, 'v6': {'nch': n6, 'pt': p6}}
        mets, boot, cis = compare(cfg, S, cent_t(S), yt, preds, gref)
        result['B_v6_test'] = {'metrics': mets, 'bootstrap': boot, 'n': len(vte), 'v6_reproduction_max_rel_dev': dev}
        pd.DataFrame({'v6_idx': vte, 'centrality': cent_t(S), 'nch_true': yt['nch'], 'nch_new': nn, 'nch_v6': n6,
                      'pt_true': yt['pt'], 'pt_new': pn, 'pt_v6': p6}).to_csv(
            os.path.join(out, 'predictions_B.csv'), index=False, float_format='%.7g')
        rep += section('B. Held-out events of the v6 sample', len(vte), notes, mets, boot)
        plots.error_vs_centrality({'new': (mets['new'], NEW), 'v6': (mets['v6'], V6)}, os.path.join(out, 'B_v6_test'),
                                  f'{cfg["label"]}: v6 test set', cis=cis)
        rep.append('![B](B_v6_test.png)\n')
    else:
        rep.append('## B. Held-out events of the v6 sample\n\nSkipped: set reference.old_data.x_file and y_file.\n')

    rep.append('Error bars in the figures: 68% bootstrap intervals of each model\'s class error. '
               'The residual metrics (β, err/σ) use this run\'s reference trend g_ref for both models.\n')
    run.write_json('compare_v6/comparison.json', result)
    with open(os.path.join(out, 'report.md'), 'w') as f:
        f.write('\n'.join(rep))
    a = result['A_new_test']['bootstrap']['nch']['class_avg']
    run.log(STEP, f'set A ({result["A_new_test"]["n"]} events): N_ch class-avg new − v6 = {100 * a["diff"]:+.2f}% '
                  f'[{100 * a["lo"]:+.2f}, {100 * a["hi"]:+.2f}]' +
                  (f'; set B ({result["B_v6_test"]["n"]} events): {100 * result["B_v6_test"]["bootstrap"]["nch"]["class_avg"]["diff"]:+.2f}%'
                   if 'B_v6_test' in result else '') + '. Read compare_v6/report.md')


if __name__ == '__main__':
    main()
