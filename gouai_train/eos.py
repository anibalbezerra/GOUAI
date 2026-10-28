"""
eos.py
------
Energy density -> entropy density with the HotQCD lattice + hadron-resonance-gas equation of state.

This is exactly the conversion used for the GOUAI v6 training images:

    s(e) = cubic interpolation of the tabulated (e, s) pairs,   s = 0 below the smallest tabulated e.

It reproduces the entropy images of the original training sample to ~3e-7 (relative, per event).
Do NOT replace it by linear interpolation or by extrapolation below the table: in dilute
regions the difference reaches 10 % per pixel, and the network would see inputs unlike
the ones it was trained on.

Table: gouai_train/data/hrg_hotqcd_eos_binary.dat, raw float64, 4 numbers per row:
       e [GeV/fm^3], p [GeV/fm^3], s [1/fm^3], T [GeV].
"""
from functools import lru_cache

import numpy as np
from scipy.interpolate import interp1d


@lru_cache(maxsize=4)
def load_eos(path):
    """Return (s_of_e, e_min): the cubic interpolator of s(e) and the smallest tabulated e."""
    d = np.fromfile(path, dtype=np.float64).reshape(-1, 4)
    return interp1d(d[:, 0], d[:, 2], kind='cubic', bounds_error=False, fill_value=0.0), float(d[0, 0])


def entropy_density(e, path):
    """Entropy-density grid(s) from energy-density grid(s) of any shape (float64 in, float64 out)."""
    s_of_e, e_min = load_eos(path)
    e = np.asarray(e, dtype=np.float64)
    s = np.zeros_like(e)
    m = e >= e_min
    s[m] = s_of_e(e[m])
    return s
