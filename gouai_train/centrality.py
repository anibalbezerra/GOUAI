"""
centrality.py
-------------
Centrality of an event from its total initial entropy (or energy) S, using a MINIMUM-BIAS table.

Centrality c (in %) = percentage of minimum-bias events with a larger S. 0 % = head-on, 100 % = grazing.

The hybrid samples used for training are not minimum bias (they are stratified, and the new
samples over-represent ultra-central events), so percentiles computed inside them are wrong:
in the original Pb-Pb sample an event at "20 %" of the sample is at 12 % of minimum bias.
The tables in reference/centrality/ come from 1.5 million TRENTo events generated with the same
parameters, grid and equation of state as the hybrid events (tools/make_centrality_table.py).

This is centrality from the INITIAL state, used here only to sort the test events into classes when
reporting accuracy. (The paper's physics results use centrality from the predicted N_ch, as in experiment.)
"""
import numpy as np
import pandas as pd


class CentralityTable:
    def __init__(self, path):
        t = pd.read_csv(path, comment='#')
        self.path = path
        self.c = t.centrality_percent.values.astype(np.float64)      # 0 ... 100, increasing
        self.S_min = t.S_min.values.astype(np.float64)               # decreasing with c

    def __call__(self, S):
        """Centrality in % for total S (array). S above the table maximum -> 0, below the minimum -> 100."""
        return np.interp(np.asarray(S, np.float64), self.S_min[::-1], self.c[::-1])


def class_labels(edges):
    return [f'{a:g}-{b:g}%' for a, b in zip(edges[:-1], edges[1:])]


def class_index(cent, edges):
    """Index of the centrality class of each event (0 = most central class); -1 outside [edges[0], edges[-1]]."""
    cent = np.asarray(cent, np.float64)
    k = np.searchsorted(np.asarray(edges, np.float64), cent, side='right') - 1
    k[cent == edges[-1]] = len(edges) - 2                  # include the upper edge in the last class
    k[(cent < edges[0]) | (cent > edges[-1])] = -1
    return k
