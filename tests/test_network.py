"""
Checks of the TensorFlow parts on tiny inputs (CPU, ~1 minute):   python -m pytest tests/test_network.py -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gouai_train import pipeline as P, model as M  # noqa: E402

TCFG = {'n_aug_mild': 8, 'n_aug_ch1': 4, 'batch_size': 16, 'head_units': 32, 'head_dropout': 0.4,
        'head_l2': 1e-4, 'trunk_dropout': 0.3, 'learning_rate': 3e-4, 'huber_delta': 1.0}


def test_pipeline_serves_13_copies_per_event():
    n, g = 5, 32
    images = np.random.default_rng(0).uniform(0, 1, (n, g, g, 2)).astype(np.float32)
    z = np.arange(2 * n, dtype=np.float32).reshape(n, 2)
    s = np.arange(n, dtype=np.float32).reshape(n, 1)
    ds = P.make_dataset(images, z, s, TCFG, ['abs', 'shape'], augment=True, seed=1)
    zs, n_plain = [], 0
    for x, zb in ds:
        assert x['image_input'].shape[1:] == (g, g, 2)
        zs.append(zb.numpy())
        for img, zz in zip(x['image_input'].numpy(), zb.numpy()):
            n_plain += np.array_equal(img, images[int(zz[0]) // 2])
    zs = np.concatenate(zs)
    assert len(zs) == 13 * n                                         # 1 + 8 + 4 copies
    assert np.array_equal(np.bincount(zs[:, 0].astype(int) // 2), [13] * n)
    assert n_plain >= n                                              # the unaugmented copies are there
    val = list(P.make_dataset(images, z, s, TCFG, ['abs', 'shape'], augment=False))
    assert np.array_equal(np.concatenate([b[1].numpy() for b in val]), z)    # validation: in order, unchanged


def test_untrained_network_outputs_zero():
    net = M.build_model(TCFG, n_channels=2, n_scalars=1, grid_n=64)
    out = M.predict_z(net, np.random.default_rng(0).uniform(0, 1, (3, 64, 64, 2)).astype(np.float32),
                      np.ones((3, 1), np.float32))
    assert out.shape == (3, 2) and np.all(out == 0)


def test_extra_inputs_change_input_shapes():
    net = M.build_model(TCFG, n_channels=3, n_scalars=2, grid_n=64)
    assert net.input['image_input'].shape[-1] == 3 and net.input['sum_input'].shape[-1] == 2
