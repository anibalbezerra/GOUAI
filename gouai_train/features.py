"""
features.py
-----------
What the network sees: the image channels and the scalar inputs. This is the place to edit when you
want to give the network something new (see docs/07_adding_inputs.md).

Image channels
--------------
A channel is named  <density>_<scaling>, e.g. entropy_abs, entropy_norm, energy_abs.
  <density>  entropy or energy: which grid of data/ it is computed from
             (energy is available only if the raw files were energy density, data.x_kind: energy)
  <scaling>  a key of SCALINGS below: how the grid x is turned into the channel

The v6 networks use two channels of the input density:
    <input>_abs  = x / 255     absolute density: how much matter is where
    <input>_norm = x / S       shape only (sums to 1): geometry independent of the overall size

Each scaling also declares its augmentation group (see pipeline.py):
    'abs'    mild flips/rotations only
    'shape'  in the "extra" augmented copies: stronger rotation plus a small random shift

Scalar inputs
-------------
features.scalars lists scalar inputs, each divided by its maximum |value| over the training events
(the normalization is stored in targets/meta.json, so predictions use the same numbers):
    S_norm         total S of the input density (the v6 choice)
    S_entropy      total entropy, S_energy  total energy (only if available)
    col:<name>     any numeric column of data/event_table.csv:
                     - a derived quantity computed from the grid in step 01 (DERIVED below, e.g.
                       col:eps2_entropy); recomputed automatically when predicting new events;
                     - a column of data.extra_table (e.g. TRENTo's b or npart) or data.extra_y_columns.
                       Careful: such a column must then also exist for every event you predict later.
"""
import numpy as np

# scaling name -> (function(x, S) -> channel, augmentation group)
#   x: (N, 280, 280) float64 grid of one density;  S: (N,) its pixel sums
SCALINGS = {
    'abs': (lambda x, S: x / 255.0, 'abs'),
    'norm': (lambda x, S: x / (S[:, None, None] + 1e-8), 'shape'),
    # add your own here, e.g.:
    # 'log': (lambda x, S: np.log1p(x), 'abs'),
}

DENSITIES = ('entropy', 'energy')


def parse_channel(name):
    density, _, scaling = name.partition('_')
    if density not in DENSITIES or scaling not in SCALINGS:
        raise ValueError(f'unknown channel {name!r}: use <density>_<scaling> with density in {DENSITIES} '
                         f'and scaling in {list(SCALINGS)}')
    return density, scaling


def densities_needed(channels):
    return sorted({parse_channel(c)[0] for c in channels})


def aug_groups(channels):
    """Augmentation group of every channel, in channel order."""
    return [SCALINGS[parse_channel(c)[1]][1] for c in channels]


def make_images(grids, sums, channels):
    """
    grids: {density: (N, 280, 280) array}   raw density grids
    sums:  {density: (N,) array}             their pixel sums S (computed in float64 from the raw grids)
    Returns the (N, 280, 280, n_channels) float32 network image.
    """
    out = []
    for c in channels:
        density, scaling = parse_channel(c)
        x = np.asarray(grids[density], np.float64)
        out.append(SCALINGS[scaling][0](x, np.asarray(sums[density], np.float64)))
    return np.stack(out, axis=-1).astype(np.float32)


# ----------------------------------------------------------------------------- derived per-event quantities
# Numbers computed from the density grids themselves in step 01 (features.derived) and stored as columns
# of the event table, so that they can be used as scalar inputs (col:<name>) or for analysis.
# Each function takes {density: (N, 280, 280) array} and returns an (N,) array. Ensemble.predict_grids
# recomputes them for new initial conditions, so a network trained with them can be used anywhere.
GRID_STEP = 0.1     # fm


def eccentricity(x, n):
    """
    Participant-plane eccentricity of order n of density grids x (N, g, g), about each grid's centroid:
        eps_n = | sum x r^n exp(i n phi) | / sum x r^n
    """
    x = np.asarray(x, np.float64)
    g = x.shape[-1]
    c = (np.arange(g) - (g - 1) / 2) * GRID_STEP
    X, Y = np.meshgrid(c, c, indexing='ij')
    w = x.sum(axis=(1, 2))
    xc = (x * X).sum(axis=(1, 2)) / w
    yc = (x * Y).sum(axis=(1, 2)) / w
    dx, dy = X[None] - xc[:, None, None], Y[None] - yc[:, None, None]
    r_n = np.hypot(dx, dy) ** n
    phase = np.exp(1j * n * np.arctan2(dy, dx))
    return np.abs((x * r_n * phase).sum(axis=(1, 2))) / (x * r_n).sum(axis=(1, 2))


DERIVED = {
    'eps2_entropy': lambda grids: eccentricity(grids['entropy'], 2),
    'eps3_entropy': lambda grids: eccentricity(grids['entropy'], 3),
    'eps2_energy': lambda grids: eccentricity(grids['energy'], 2),
    # add your own here: 'name': lambda grids: <(N,) array from grids['entropy'] / grids['energy']>
}


def derived_needed(cfg):
    """Derived quantities to compute in step 01: features.derived plus any col:<name> scalar that is derived."""
    names = list(cfg['features'].get('derived') or [])
    names += [s[4:] for s in cfg['features']['scalars'] if s.startswith('col:') and s[4:] in DERIVED]
    unknown = [n for n in names if n not in DERIVED]
    if unknown:
        raise ValueError(f'unknown derived quantities {unknown}; available: {list(DERIVED)}')
    return list(dict.fromkeys(names))


def scalar_raw(name, table, input_density):
    """Un-normalized values of one scalar input for the rows of `table` (a DataFrame)."""
    if name == 'S_norm':
        return table[f'S_{input_density}'].values.astype(np.float64)
    if name in ('S_entropy', 'S_energy'):
        return table[name].values.astype(np.float64)
    if name.startswith('col:'):
        col = name[4:]
        if col not in table:
            raise KeyError(f'scalar input {name}: column {col!r} is not in the event table')
        return table[col].values.astype(np.float64)
    raise ValueError(f'unknown scalar input {name!r} (see features.py)')


def scalar_norms(names, table, input_density):
    """Normalization constant (max |value| over `table`, normally the training events) of each scalar."""
    return {n: float(np.max(np.abs(scalar_raw(n, table, input_density)))) for n in names}


def make_scalars(names, norms, table, input_density):
    """(N, n_scalars) float32 matrix of normalized scalar inputs."""
    cols = [scalar_raw(n, table, input_density) / norms[n] for n in names]
    return np.stack(cols, axis=1).astype(np.float32)
