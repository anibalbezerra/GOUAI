"""
Fast checks of the numpy-only parts (no TensorFlow, no data files):   python -m pytest tests/ -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gouai_train import config as C, splits as SP, features as F, metrics as MT, rawdata, eos  # noqa: E402
from gouai_train.targets import ResidualSpec, Trend  # noqa: E402
from gouai_train.centrality import CentralityTable, class_index  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def synthetic(n=300, seed=0):
    rng = np.random.default_rng(seed)
    S = np.exp(rng.uniform(np.log(2e3), np.log(1.2e6), n))
    nch = 1.2e-3 * S ** 1.0 * np.exp(rng.normal(0, 0.1, n))
    pt = 0.3 * S ** 0.05 * np.exp(rng.normal(0, 0.04, n))
    return S, nch, pt


# ----------------------------------------------------------------------------- targets
def test_residual_round_trip():
    S, nch, pt = synthetic()
    spec = ResidualSpec.fit(S, nch, pt, 3)
    z = spec.transform(S, nch, pt)
    assert np.allclose(z.mean(0), 0, atol=1e-10) and np.allclose(z.std(0, ddof=1), 1)
    n2, p2 = spec.inverse(S, z)
    assert np.allclose(n2, nch, rtol=1e-12) and np.allclose(p2, pt, rtol=1e-12)
    spec2 = ResidualSpec.from_dict(spec.to_dict())
    assert np.allclose(spec2.transform(S, nch, pt), z)


def test_trend_linear_continuation():
    S, nch, _ = synthetic()
    t = Trend.fit(S, nch, 3)
    S_out = np.array([S.max() * 2, S.max() * 4])
    lg = t.log_g(S_out)
    slope = (lg[1] - lg[0]) / np.log(2)
    edge_slope = np.polyval(np.polyder(t.coef), t.u_max)
    assert np.isclose(slope, edge_slope)


# ----------------------------------------------------------------------------- splits
def test_outer_split_and_folds():
    S, _, _ = synthetic(400)
    forced = [3, 7, 11]
    tr, te = SP.outer_split(S, 0.25, 42, 10, forced)
    assert set(forced) <= set(te) and not set(tr) & set(te) and len(tr) + len(te) == 400
    assert abs(len(te) - 100) <= 1
    split = {'outer_train': tr.tolist(), 'outer_test': te.tolist(), 'folds': SP.make_folds(tr, S, 5, 42, 10)}
    SP.check_split(split, 400)
    sub = SP.subsample(tr, S, 120, 0, 10)
    assert len(sub) == 120 and set(sub) <= set(tr)


def test_split_stratification_keeps_central_events_in_test():
    S, _, _ = synthetic(1000)
    S[:200] = np.linspace(1.1e6, 1.2e6, 200)              # an over-represented ultra-central group
    tr, te = SP.outer_split(S, 0.25, 1, 10)
    frac = np.mean(np.isin(np.arange(200), te))
    assert 0.18 < frac < 0.32


def test_match_events():
    a_n, a_p = np.array([10.0, 20.0, 30.0]), np.array([0.5, 0.6, 0.7])
    b_n, b_p = np.array([30.0, 5.0, 10.0]), np.array([0.7, 0.4, 0.5])
    assert SP.match_events(a_n, a_p, b_n, b_p).tolist() == [2, -1, 0]


# ----------------------------------------------------------------------------- centrality
def test_centrality_table():
    for system in ('2.76_PbPb', '0.2_AuAu'):
        for inp in ('entropy', 'energy'):
            t = CentralityTable(os.path.join(REPO, 'reference', 'centrality', f'{system}_{inp}.csv'))
            assert np.all(np.diff(t.S_min) <= 0)
            assert np.isclose(t(t.S_min[1000]), 10.0, atol=0.02)          # row 1000 = 10.00 %
            assert t(1e12) == 0 and t(-1) == 100
    k = class_index(np.array([0.0, 0.5, 1.0, 99.9, 100.0, 101.0]), [0, 1, 5, 100])
    assert k.tolist() == [0, 0, 1, 2, 2, -1]


# ----------------------------------------------------------------------------- features
def test_features_default_channels():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 50, (3, 280, 280))
    S = x.reshape(3, -1).sum(1)
    img = F.make_images({'entropy': x}, {'entropy': S}, ['entropy_abs', 'entropy_norm'])
    assert img.shape == (3, 280, 280, 2) and img.dtype == np.float32
    assert np.allclose(img[..., 0], x / 255.0, rtol=1e-6)
    assert np.allclose(img[..., 1].reshape(3, -1).sum(1), 1.0, rtol=1e-5)
    assert F.aug_groups(['entropy_abs', 'entropy_norm']) == ['abs', 'shape']
    with pytest.raises(ValueError):
        F.parse_channel('entropy_unknown')


def test_scalars():
    tab = pd.DataFrame({'S_entropy': [1.0, 2.0, 4.0], 'b': [3.0, 6.0, 12.0]})
    norms = F.scalar_norms(['S_norm', 'col:b'], tab, 'entropy')
    sc = F.make_scalars(['S_norm', 'col:b'], norms, tab, 'entropy')
    assert np.allclose(sc, [[0.25, 0.25], [0.5, 0.5], [1, 1]])


# ----------------------------------------------------------------------------- eos
def test_eos_zero_below_table():
    path = os.path.join(REPO, 'gouai_train', 'data', 'hrg_hotqcd_eos_binary.dat')
    s_of_e, e_min = eos.load_eos(path)
    s = eos.entropy_density(np.array([0.0, e_min / 2, 1.0, 10.0]), path)
    assert s[0] == 0 and s[1] == 0 and 0 < s[2] < s[3]


# ----------------------------------------------------------------------------- raw data
def test_rawdata_layouts(tmp_path):
    n_pix = 280 * 280
    rng = np.random.default_rng(2)
    grids = rng.uniform(0, 1, (3, n_pix))
    ids = [5, 9, 12]
    y = tmp_path / 'Y.dat'
    np.savetxt(y, np.c_[ids, [10, 20, 30], [0.5, 0.6, 0.7]])
    xa = tmp_path / 'x.dat'
    np.savetxt(xa, grids)
    xb = tmp_path / 'xdir'
    xb.mkdir()
    for eid, g in zip(ids[::-1], grids[::-1]):                           # written in another order
        np.savetxt(xb / f'flattened_IC_{eid}.txt', g)
    np.savetxt(xb / 'flattened_IC_77.txt', grids[0])                      # a file without labels
    for x_path in (xa, xb):
        cfg = {'data': {'x_path': str(x_path), 'y_file': str(y), 'x_pattern': r'(\d+)', 'grid_n': 280,
                        'y_columns': {'id': 0, 'nch': 1, 'pt': 2}, 'extra_y_columns': {}}}
        labels = rawdata.read_labels(cfg)
        has, notes = rawdata.match_report(cfg, labels)
        assert has.all()
        got = dict(rawdata.iter_events(cfg, labels))
        for row in range(3):
            assert np.allclose(got[row], grids[row])
    assert any('NO label' in n for n in notes)


# ----------------------------------------------------------------------------- config
def test_config_rejects_linked_dirs_and_merges(tmp_path):
    p = tmp_path / 'c.yaml'
    p.write_text(yaml.safe_dump({'system': '2.76_PbPb', 'run_dir': '/home/CHANGE/run'}))
    with pytest.raises(C.ConfigError, match='CHANGE'):
        C.load_config(p)
    links = [d for d in os.listdir('/') if os.path.islink('/' + d) and os.path.isdir('/' + d)]
    if links:                                    # e.g. /scratch -> /home/scratch, /bin -> usr/bin
        assert C.linked_top_dir(f'/{links[0]}/x/run')
    assert C.linked_top_dir(str(tmp_path)) == '' or os.path.islink('/' + str(tmp_path).split('/')[1])
    p.write_text(yaml.safe_dump({'system': '0.2_AuAu', 'run_dir': str(tmp_path / 'run'),
                                 'training': {'epochs': 5}}))
    cfg = C.load_config(p)
    assert cfg['training']['epochs'] == 5 and cfg['training']['batch_size'] == 256
    assert cfg['features']['channels'] == ['entropy_abs', 'entropy_norm']
    assert cfg['evaluation']['centrality_table'].endswith('0.2_AuAu_entropy.csv')


def test_example_configs_load(tmp_path):
    for name in os.listdir(os.path.join(REPO, 'configs', 'examples')):
        src = os.path.join(REPO, 'configs', 'examples', name)
        with pytest.raises(C.ConfigError, match='CHANGE'):           # templates must be edited first
            C.load_config(src)
        dst = tmp_path / name
        dst.write_text(open(src).read().replace('CHANGE_USER', 'student').replace('CHANGE', 'x'))
        C.load_config(dst)


# ----------------------------------------------------------------------------- metrics
def test_metrics_and_class_average():
    cfg = C.resolve(C.deep_merge(yaml.safe_load(open(C.DEFAULTS)),
                                 {'system': '2.76_PbPb', 'run_dir': '/home/scratch/student/run'}))
    S, nch, pt = synthetic(500)
    spec = ResidualSpec.fit(S, nch, pt, 3)
    cent = np.linspace(0, 100, 500)
    pred = {'nch': nch * 1.02, 'pt': pt}
    m = MT.evaluate(S, {'nch': nch, 'pt': pt}, pred, spec, cent, cfg)
    assert np.isclose(m['nch']['all']['mean_rel_err'], 0.02)
    assert np.isclose(m['nch']['class_avg']['mean_rel_err'], 0.02)
    assert m['pt']['all']['mae'] == 0
    b = MT.paired_bootstrap(np.full(50, 0.02), np.full(50, 0.03), 200, 0)
    assert np.isclose(b['diff'], -0.01) and b['hi'] < 0


def test_eccentricity_of_elliptic_gaussian():
    c = (np.arange(280) - 139.5) * 0.1
    X, Y = np.meshgrid(c, c, indexing='ij')
    sx, sy = 1.5, 2.5
    g = np.exp(-(X - 1) ** 2 / (2 * sx ** 2) - (Y + 2) ** 2 / (2 * sy ** 2))[None]    # off-centre on purpose
    assert np.isclose(F.eccentricity(g, 2)[0], (sy ** 2 - sx ** 2) / (sx ** 2 + sy ** 2), rtol=1e-4)   # 0.1 fm grid
    assert F.eccentricity(g, 3)[0] < 1e-3
