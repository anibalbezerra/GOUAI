"""
training.py
-----------
Training one network (used by steps 04 smoke test, 05 cross-validation folds, 09 final model)
and turning its output into physical predictions.

The schedule (training section of the config, identical to v6):
  * Adam, learning rate 3e-4, batch 256, Huber loss on the two standardized residuals.
  * WARM-UP, the first 100 epochs: no early stopping, no learning-rate cuts, no checkpoint.
    In every v6 fold the validation loss jumped 15-45x after epoch 1 and needed 60-150 epochs
    to settle (most likely the BatchNorm statistics of a ResNet trained from scratch on few events).
    Without the warm-up one fold was stopped with its epoch-1 weights.
  * After the warm-up: learning rate x 0.5 after 15 epochs without improvement of the validation
    loss (down to 1e-6); stop after 80 epochs without improvement; keep the weights of the best
    epoch (WarmupCheckpoint) and load them at the end.
  * The final model (step 09) has no validation set: it trains for a fixed number of epochs,
    the median best epoch of the 5 folds.

Checkpoints are written to node-local disk and copied to the run directory at the end:
HDF5 file locking on a network file system (NFS) can hang the process.
"""
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

from . import gpu_env  # noqa: F401
import tensorflow as tf  # noqa: E402

from . import model as M  # noqa: E402
from . import pipeline as P  # noqa: E402
from . import features as F  # noqa: E402


# ----------------------------------------------------------------------------- callbacks
class WarmupCheckpoint(tf.keras.callbacks.Callback):
    """Save the weights of the best epoch, considering only epochs after the warm-up."""

    def __init__(self, path, monitor, warmup):
        super().__init__()
        self.path, self.monitor, self.warmup = path, monitor, warmup
        self.best = np.inf
        self.best_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        v = (logs or {}).get(self.monitor)
        if epoch + 1 > self.warmup and v is not None and v < self.best:
            self.best, self.best_epoch = v, epoch + 1
            self.model.save_weights(self.path)


class WarmupReduceLR(tf.keras.callbacks.ReduceLROnPlateau):
    """ReduceLROnPlateau that ignores the warm-up epochs."""

    def __init__(self, warmup, **kw):
        super().__init__(**kw)
        self.warmup = warmup

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 > self.warmup:
            super().on_epoch_end(epoch, logs)


# ----------------------------------------------------------------------------- environment
def require_gpu(run, step, allow_cpu=False):
    """Stop at once if TensorFlow sees no GPU (training on CPU is ~100x slower and would time out)."""
    gpus = tf.config.list_physical_devices('GPU')
    print('GPUs visible to TensorFlow:', gpus, flush=True)
    if not gpus and not allow_cpu:
        run.log(step, f'NO GPU visible on {os.uname().nodename}; stopped (see docs/08_troubleshooting.md)')
        sys.exit(3)
    return gpus


def local_workdir(run, tag):
    base = os.environ.get('SLURM_TMPDIR') or os.path.join('/tmp', os.environ.get('USER', 'user'))
    job = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
    path = os.path.join(base, f'gouai_{job}', tag)
    os.makedirs(path, exist_ok=True)
    return path


# ----------------------------------------------------------------------------- data for one network
def arrays(cfg, store, spec, norms, idx):
    """Images, standardized targets z and normalized scalars for the events idx."""
    idx = np.asarray(idx)
    S = store.S(idx)
    y = store.y(idx)
    z = spec.transform(S, y['nch'], y['pt'])
    return store.images(idx), z, store.scalars(idx, norms)


# ----------------------------------------------------------------------------- training
def fit_network(run, store, spec, norms, train_idx, val_idx, out_dir, epochs, seed, warmup=None):
    """
    Train one network. With val_idx: early stopping and best-epoch selection on the validation
    events (cross-validation folds). Without: a fixed number of epochs (final model).
    Writes out_dir/history.csv and out_dir/best.weights.h5. Returns (model, info).
    """
    cfg = run.cfg
    tcfg = cfg['training']
    warmup = tcfg.get('warmup_epochs', 0) if warmup is None else warmup
    tf.keras.utils.set_random_seed(seed)
    groups = F.aug_groups(cfg['features']['channels'])

    x_tr, z_tr, s_tr = arrays(cfg, store, spec, norms, train_idx)
    ds_tr = P.make_dataset(x_tr, z_tr, s_tr, tcfg, groups, augment=True, seed=tcfg['seed'])
    ds_va = None
    if val_idx is not None:
        x_va, z_va, s_va = arrays(cfg, store, spec, norms, val_idx)
        ds_va = P.make_dataset(x_va, z_va, s_va, tcfg, groups, augment=False)

    model = M.compile_model(M.build_for_run(cfg), tcfg)
    local = local_workdir(run, os.path.relpath(out_dir, run.dir).replace(os.sep, '_'))
    weights = os.path.join(local, 'best.weights.h5')
    monitor = 'val_loss' if ds_va is not None else 'loss'
    callbacks = [WarmupReduceLR(warmup, monitor=monitor, factor=tcfg['rlr_factor'],
                                patience=tcfg['rlr_patience'], min_lr=tcfg['min_lr']),
                 tf.keras.callbacks.CSVLogger(os.path.join(out_dir, 'history.csv'))]
    ckpt = None
    if ds_va is not None:
        ckpt = WarmupCheckpoint(weights, 'val_loss', warmup)
        callbacks += [ckpt, tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=tcfg['es_patience'],
                                                             start_from_epoch=warmup)]
    t0 = time.time()
    hist = model.fit(ds_tr, validation_data=ds_va, epochs=epochs, callbacks=callbacks, verbose=2)
    info = {'epochs_run': len(hist.history['loss']), 'wall_s': time.time() - t0, 'warmup_epochs': warmup,
            'n_train': len(train_idx), 'n_val': 0 if val_idx is None else len(val_idx), 'seed': seed}
    if ds_va is not None:
        vl = np.array(hist.history['val_loss'])
        if ckpt.best_epoch is None:
            raise RuntimeError('training ended before the warm-up was over: no epoch could be selected')
        info.update({'best_epoch': ckpt.best_epoch, 'best_val_loss': float(ckpt.best),
                     'val_loss_epoch1': float(vl[0]), 'val_loss_last': float(vl[-1])})
        model.load_weights(weights)
    else:
        model.save_weights(weights)
    shutil.copyfile(weights, os.path.join(out_dir, 'best.weights.h5'))
    shutil.rmtree(local, ignore_errors=True)
    return model, info


# ----------------------------------------------------------------------------- predictions
def predict_physical(model, cfg, store, spec, norms, idx, chunk=512):
    """N_ch, [pT] (GeV) and z for the events idx, computed in chunks to bound memory."""
    idx = np.asarray(idx)
    zs = []
    for i in range(0, len(idx), chunk):
        part = idx[i:i + chunk]
        zs.append(M.predict_z(model, store.images(part), store.scalars(part, norms),
                              batch_size=cfg['training']['predict_batch']))
    z = np.concatenate(zs) if zs else np.zeros((0, 2))
    nch, pt = spec.inverse(store.S(idx), z)
    return nch, pt, z


def prediction_table(model, cfg, store, spec, norms, idx, part):
    """One row per event: truth, prediction and z, tagged with `part` ('oof', 'test', ...)."""
    idx = np.asarray(idx)
    nch, pt, z = predict_physical(model, cfg, store, spec, norms, idx)
    y = store.y(idx)
    return pd.DataFrame({'idx': idx, 'part': part, 'nch_true': y['nch'], 'pt_true': y['pt'],
                         'nch_pred': nch, 'pt_pred': pt, 'z_nch': z[:, 0], 'z_pt': z[:, 1],
                         'S': store.S(idx)})
