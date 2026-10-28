#!/usr/bin/env python3
"""
make_centrality_table.py
------------------------
Build a minimum-bias centrality table: the total initial entropy (or energy) S that
bounds each centrality percentile, from a large minimum-bias sample of TRENTo events.

    python tools/make_centrality_table.py --events events.csv.gz --column S_entropy \
        --label "Pb-Pb 2.76 TeV, entropy" --out reference/centrality/2.76_PbPb_entropy.csv

The output has one row per 0.01% of centrality:
    centrality_percent, S_min
meaning: an event with total S >= S_min belongs to the most central `centrality_percent` %
of minimum-bias events. Centrality 0% = largest S (head-on), 100% = smallest S.

Why a minimum-bias reference is needed
--------------------------------------
The hybrid training samples are NOT minimum bias: they are stratified by centrality, and the
new samples deliberately contain many more ultra-central events. Percentiles computed inside
such a sample are biased (if half of the events are in 0-5%, the "median" event of the sample
is a 5% event, not a 50% one). Centrality must therefore be read from a minimum-bias sample
generated with the same TRENTo parameters, grid and equation of state.

The tables shipped in reference/centrality/ were built from the 1.5 x 10^6-event TRENTo samples
of the published GOUAI v6 production (columns S_entropy / S_energy of events.csv.gz).
Any other minimum-bias sample works: the input only needs one column with the pixel sum S.
"""
import argparse
import datetime
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--events', required=True, help='CSV (optionally .gz) with one row per minimum-bias event')
    ap.add_argument('--column', required=True, help='column holding the pixel sum S of the input density')
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--step', type=float, default=0.01, help='centrality resolution in percent')
    a = ap.parse_args()

    S = pd.read_csv(a.events, usecols=[a.column])[a.column].values.astype(np.float64)
    c = np.round(np.arange(0.0, 100.0 + a.step / 2, a.step), 6)
    # the c% most central events are those with S above the (100 - c)% quantile
    S_min = np.quantile(S, 1.0 - c / 100.0)
    header = (f'# minimum-bias centrality table: {a.label}\n'
              f'# source: {os.path.basename(a.events)} (column {a.column}, {len(S)} events), built {datetime.date.today()}\n'
              f'# an event with S >= S_min is in the most central centrality_percent %\n')
    with open(a.out, 'w') as f:
        f.write(header)
        pd.DataFrame({'centrality_percent': c, 'S_min': S_min}).to_csv(f, index=False, float_format='%.8g')
    print(f'wrote {a.out}: {len(S)} events, S range [{S.min():.4g}, {S.max():.4g}]')


if __name__ == '__main__':
    main()
