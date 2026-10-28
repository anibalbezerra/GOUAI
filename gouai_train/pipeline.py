"""
pipeline.py
-----------
The tf.data input pipeline and the data augmentation used during training.

Data augmentation (identical to the v6 training)
------------------------------------------------
N_ch and [pT] do not change when the initial condition is flipped or rotated (the observables are
integrated over azimuth), so flipped/rotated copies of a training event are new, equally valid
training examples. With only a few hundred events this is essential against overfitting.
In every epoch each training event appears 1 + n_aug_mild + n_aug_ch1 = 13 times:

    1 x            unchanged
    n_aug_mild x   random left-right flip, up-down flip, rotation by up to +-18 deg (factor 0.05 of a turn);
                   all channels transformed together
    n_aug_ch1 x    each channel transformed INDEPENDENTLY:
                     'abs' channels  : flips + rotation up to +-18 deg
                     'shape' channels: flips + rotation up to +-54 deg (factor 0.15), then a random
                                       shift of up to 14 pixels (reflect padding + random 280x280 crop)

The 13 x N examples are fully shuffled every epoch and served in batches of 256.

How it is implemented (and why it differs from the v6 code, with the same result)
---------------------------------------------------------------------------------
v6 built the 13 copies as 13 concatenated datasets of images and shuffled them with a 20000-element
buffer. With ~450 events that is a full shuffle of 5850 images (3.7 GB in the buffer). With several
thousand events the buffer would no longer cover an epoch (the shuffle becomes partial) and would
need tens of GB. Here only the (event, copy-type) INDEX pairs are shuffled; the image is fetched
from memory and augmented when the batch is built. For the v6 sample size this is the same
procedure; for larger samples it stays a full shuffle at a fixed, small memory cost.
"""
import numpy as np

from . import gpu_env  # noqa: F401
import tensorflow as tf  # noqa: E402

COPY_PLAIN, COPY_MILD, COPY_EXTRA = 0, 1, 2

_mild_rotation = tf.keras.layers.RandomRotation(factor=0.05)
_strong_rotation = tf.keras.layers.RandomRotation(factor=0.15)
SHIFT_PAD = 14


def _flips(x):
    x = tf.image.random_flip_left_right(x)
    return tf.image.random_flip_up_down(x)


def augment_mild(image):
    """All channels together: flips + rotation up to +-18 deg."""
    return _mild_rotation(_flips(image), training=True)


def make_augment_extra(groups, grid_n):
    """Per-channel independent augmentation; `groups` = features.aug_groups(channels)."""
    def augment_extra(image):
        chans = []
        for c, group in enumerate(groups):
            x = _flips(image[:, :, c:c + 1])
            if group == 'shape':
                x = _strong_rotation(x, training=True)
                x = tf.pad(x, [[SHIFT_PAD, SHIFT_PAD], [SHIFT_PAD, SHIFT_PAD], [0, 0]], mode='REFLECT')
                x = tf.image.random_crop(x, size=(grid_n, grid_n, 1))
            else:
                x = _mild_rotation(x, training=True)
            chans.append(x)
        return tf.concat(chans, axis=-1)
    return augment_extra


def make_dataset(images, z, scalars, tcfg, groups, augment, seed=None):
    """
    images  : (N, H, W, C) float32 numpy array, in memory
    z       : (N, 2) standardized targets
    scalars : (N, n_scalars) normalized scalar inputs
    augment : True for training (13 shuffled copies per event), False for validation (as is, in order)
    Returns a dataset of ({'image_input', 'sum_input'}, z) batches.
    """
    n, h, w, c = images.shape
    images = np.ascontiguousarray(images, dtype=np.float32)
    z_t = tf.constant(np.asarray(z, np.float32))
    s_t = tf.constant(np.asarray(scalars, np.float32))

    if augment:
        kinds = [COPY_PLAIN] + [COPY_MILD] * tcfg['n_aug_mild'] + [COPY_EXTRA] * tcfg['n_aug_ch1']
        rows = np.tile(np.arange(n), len(kinds))
        kind = np.repeat(kinds, n)
        ds = tf.data.Dataset.from_tensor_slices((rows.astype(np.int64), kind.astype(np.int32)))
        ds = ds.shuffle(len(rows), seed=seed, reshuffle_each_iteration=True)
    else:
        ds = tf.data.Dataset.from_tensor_slices((np.arange(n, dtype=np.int64), np.zeros(n, np.int32)))

    augment_extra = make_augment_extra(groups, h)

    def fetch(row):
        return images[row]

    def load(row, k):
        img = tf.numpy_function(fetch, [row], tf.float32)
        img.set_shape((h, w, c))
        if augment:
            img = tf.switch_case(k, branch_fns=[lambda: img,
                                                lambda: augment_mild(img),
                                                lambda: augment_extra(img)])
        return {'image_input': img, 'sum_input': tf.gather(s_t, row)}, tf.gather(z_t, row)

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(tcfg['batch_size']).prefetch(tf.data.AUTOTUNE)
