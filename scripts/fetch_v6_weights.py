#!/usr/bin/env python3
"""
fetch_v6_weights.py - put the published GOUAI v6 network weights where step 08 expects them.

The weights (20 files of 135 MB) are too large for git. They are distributed with the GOUAI
release (Zenodo record of the paper). This script places them in
    reference/v6/<system>/<input>_pt/fold_<k>/model.weights.h5
and checks every file against its SHA-256 checksum (reference/v6/weights_manifest.json).

    python scripts/fetch_v6_weights.py --from-dir /path/to/unpacked/zenodo/weights      (symlinks, no copy)
    python scripts/fetch_v6_weights.py --from-dir ... --copy                            (copies)
    python scripts/fetch_v6_weights.py --zenodo RECORD_ID                               (downloads)
    python scripts/fetch_v6_weights.py --only 2.76_PbPb/entropy_pt                     (one model only)

Inside the lab, the folder with the files is given in the lab notes.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V6 = os.path.join(REPO, 'reference', 'v6')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--from-dir')
    ap.add_argument('--zenodo')
    ap.add_argument('--copy', action='store_true')
    ap.add_argument('--only', default=None, help='e.g. 2.76_PbPb/entropy_pt')
    a = ap.parse_args()
    if not (a.from_dir or a.zenodo):
        sys.exit('give --from-dir or --zenodo')
    manifest = json.load(open(os.path.join(V6, 'weights_manifest.json')))
    n_ok = 0
    for dest_rel, info in manifest.items():
        if a.only and not dest_rel.startswith(a.only):
            continue
        dest = os.path.join(V6, dest_rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest) and sha256(dest) == info['sha256']:
            n_ok += 1
            continue
        if os.path.lexists(dest):
            os.remove(dest)
        if a.from_dir:
            src = os.path.join(os.path.abspath(a.from_dir), info['zenodo_file'])
            if not os.path.exists(src):
                sys.exit(f'{src} not found')
            if a.copy:
                shutil.copyfile(src, dest)
            else:
                os.symlink(src, dest)
        else:
            url = f'https://zenodo.org/records/{a.zenodo}/files/{info["zenodo_file"]}?download=1'
            print('downloading', url, flush=True)
            urllib.request.urlretrieve(url, dest)
        if sha256(dest) != info['sha256']:
            os.remove(dest)
            sys.exit(f'checksum mismatch for {info["zenodo_file"]}: file removed')
        print('ok', dest_rel)
        n_ok += 1
    print(f'{n_ok} weight files in place and verified')


if __name__ == '__main__':
    main()
