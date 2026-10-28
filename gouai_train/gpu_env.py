"""
gpu_env.py
----------
Import this module BEFORE tensorflow (the package does it for you in model.py).

It loads the CUDA libraries that pip installs together with tensorflow[and-cuda]
(site-packages/nvidia/*/lib) with RTLD_GLOBAL, so that TensorFlow finds them even when the
node's own CUDA installation lacks one of them.

Why: on one GPU node of our cluster libcusolver.so.11 was missing from the system; TensorFlow then
printed "Cannot dlopen some GPU libraries", registered no GPU and silently trained on the CPU
(about 100x slower). LD_LIBRARY_PATH cannot be changed from inside a running process, hence the preload.
The training scripts also stop at once if no GPU is visible (see training.require_gpu).
"""
import ctypes
import glob
import os
import sys

# dependency order: each library's own dependencies are loaded before it
_ORDER = ['nvjitlink', 'cuda_runtime', 'cublas', 'cusparse', 'cusolver', 'cufft', 'curand', 'cudnn']


def preload():
    site = next((p for p in sys.path if p.endswith('site-packages') and os.path.isdir(os.path.join(p, 'nvidia'))), None)
    if site is None:
        return []
    loaded = []
    for pkg in _ORDER:
        for lib in sorted(glob.glob(os.path.join(site, 'nvidia', pkg, 'lib', 'lib*.so.*'))):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                loaded.append(os.path.basename(lib))
            except OSError:
                pass
    return loaded


preload()
