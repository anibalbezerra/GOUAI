"""
splits.py
---------
Which events are used for what. All indices refer to the row `idx` of <run_dir>/data/event_table.csv.

    outer_test   held-out events: used ONLY in step 07 to measure the final accuracy.
                 Never used for training, early stopping, or choosing anything.
    outer_train  everything else. It is divided into n_folds (5) folds:
    folds[k]     train = 4/5 of outer_train, val = the remaining 1/5.
                 Network k is trained on `train` and stopped early on `val`.
                 Every outer_train event is in exactly one `val`, so every training event gets one
                 "out-of-fold" (OOF) prediction from a network that never saw it (step 06).

Both the held-out split and the folds are stratified on S (total initial entropy): events are sorted
into n_strata quantile bins of S and each bin is divided in the same proportion. This keeps all
centralities, including the ultra-central events, represented in every part.

Protection of the v6 held-out events (splits.protect_reference_test)
--------------------------------------------------------------------
If the new sample contains events of the old (v6) sample, the events that v6 held out are forced
into the new held-out set. They therefore stay unseen by both the old and the new networks, and
step 08 can compare the two on exactly the events v6 was validated on. Events are recognized by
their labels: N_ch and [pT] equal to 1e-5 relative precision.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def s_strata(S, n_strata):
    """Quantile bin (0 ... n_strata-1) of every S."""
    qs = np.quantile(S, np.linspace(0, 1, n_strata + 1)[1:-1])
    return np.digitize(S, qs)


def outer_split(S, test_fraction, seed, n_strata, forced_test=()):
    """
    Held-out split stratified on S. Events in `forced_test` always go to the test set;
    the others are split so that the total test fraction is about `test_fraction`.
    """
    n = len(S)
    forced = np.array(sorted(set(int(i) for i in forced_test)), dtype=int)
    rest = np.setdiff1d(np.arange(n), forced)
    n_test_rest = max(int(round(test_fraction * n)) - len(forced), 0)
    if n_test_rest == 0:
        return np.sort(rest), np.sort(forced)
    tr, te = train_test_split(rest, test_size=n_test_rest, random_state=seed,
                              stratify=s_strata(S[rest], n_strata))
    return np.sort(tr), np.sort(np.concatenate([te, forced]))


def make_folds(train_idx, S, n_folds, seed, n_strata):
    """Stratified K-fold over the outer-train events (same algorithm as for the v6 models)."""
    train_idx = np.asarray(train_idx)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for k, (a, b) in enumerate(skf.split(train_idx, s_strata(S[train_idx], n_strata))):
        folds.append({'fold': k, 'train': np.sort(train_idx[a]).tolist(), 'val': np.sort(train_idx[b]).tolist()})
    return folds


def subsample(train_idx, S, n, seed, n_strata):
    """Keep n of the outer-train events, stratified on S (learning-curve experiments)."""
    train_idx = np.asarray(train_idx)
    if n >= len(train_idx):
        return np.sort(train_idx)
    keep, _ = train_test_split(train_idx, train_size=n, random_state=seed,
                               stratify=s_strata(S[train_idx], n_strata))
    return np.sort(keep)


def check_split(split, n_events=None):
    """Assert that the split is consistent: no overlaps, folds cover outer_train exactly."""
    tr, te = set(split['outer_train']), set(split['outer_test'])
    assert not (tr & te), 'outer train/test overlap'
    seen = set()
    for f in split['folds']:
        a, b = set(f['train']), set(f['val'])
        assert not (a & b), f'fold {f["fold"]}: train/val overlap'
        assert (a | b) == tr, f'fold {f["fold"]}: does not cover outer train'
        assert not ((a | b) & te), f'fold {f["fold"]}: touches outer test'
        assert not (b & seen), 'validation sets of two folds overlap'
        seen |= b
    assert seen == tr, 'validation sets do not cover outer train'
    if n_events is not None:
        assert max(tr | te) < n_events, 'split refers to events that do not exist'


def match_events(nch_a, pt_a, nch_b, pt_b, rtol=1e-5):
    """
    For every event of sample A, the index of the identical event of sample B (same N_ch and [pT]
    to rtol), or -1. Raises if one event of A matches two events of B.
    """
    nch_a, pt_a = np.asarray(nch_a, float), np.asarray(pt_a, float)
    nch_b, pt_b = np.asarray(nch_b, float), np.asarray(pt_b, float)
    out = np.full(len(nch_a), -1, dtype=int)
    for i in range(0, len(nch_a), 2000):                     # blocks, to bound memory
        dn = np.abs(nch_a[i:i + 2000, None] / nch_b[None, :] - 1)
        dp = np.abs(pt_a[i:i + 2000, None] / pt_b[None, :] - 1)
        ok = (dn < rtol) & (dp < rtol)
        n_match = ok.sum(1)
        if (n_match > 1).any():
            raise ValueError('an event matches several reference events: labels are not unique enough to match')
        hit = n_match == 1
        out[i:i + 2000][hit] = ok[hit].argmax(1)
    return out
