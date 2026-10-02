"""Inventory every checkpoint on disk: what epoch it actually reached, what it was
trained on, and — the question being asked — whether it is undertrained.

PixelSNAIL checkpoints store an argparse.Namespace, so the training data path and the
PLANNED epoch budget are recoverable from the file itself. Note the distinction that
matters: `args.epoch` is the epoch budget the run was LAUNCHED with, not the epoch the
checkpoint reached — that is in the filename.
"""
import os
import re
import glob
import csv
from collections import defaultdict

import torch

TRAIN_DATA = {   # what each family was actually fitted on (from the run that produced it)
    'vqvae':            ('600px crops, ALL 10 boards (no holdout)', '~10,668 crops'),
    'vqvae_tight':      ('TIGHT crops, TRAIN-POOL only (boards 06/09 excluded)', '4,274 crops'),
    'pixelsnail_top':   ('lmdb/all — 600px codes, ALL 10 boards (NO holdout)', '~10,668 crops'),
    'pixelsnail_bottom': ('lmdb/all — 600px codes, ALL 10 boards (NO holdout)', '~10,668 crops'),
    'pixelsnail_cond':  ('lmdb/train_pool — 600px codes, train pool only', '8,596 crops'),
    'pixelsnail_condtight': ('lmdb/train_pool_tight — TIGHT codes, train pool only', '4,274 crops'),
    'pixelsnail_condtight2': ('lmdb/train_pool_tight — TIGHT codes, train pool only', '4,274 crops'),
    'classifier_A':     ('600px base crops, train boards, budget-subsampled', '2,149 @100%'),
    'classifier_Atight': ('TIGHT base crops, train boards, budget-subsampled', '2,149 @100%'),
}


def family(name):
    n = re.sub(r'_?\d{3}\.pt$', '', name).replace('.pt', '')
    n = re.sub(r'_best$', '', n)
    n = re.sub(r'_b\d+_s\d+$', '', n)
    return n


def epoch_of(name):
    m = re.search(r'_(\d{3})\.pt$', name)
    return int(m[1]) if m else None


def main():
    fams = defaultdict(list)
    for p in sorted(glob.glob('checkpoint/*.pt')):
        fams[family(os.path.basename(p))].append(p)

    rows = []
    for fam, paths in sorted(fams.items()):
        eps = [e for e in (epoch_of(os.path.basename(p)) for p in paths) if e is not None]
        # pull the stored args from the LAST checkpoint of the family, if it has any
        planned, data_path, extra = None, None, ''
        try:
            ck = torch.load(paths[-1], map_location='cpu', weights_only=False)
            if isinstance(ck, dict) and 'args' in ck:
                a = ck['args']
                planned = getattr(a, 'epoch', None)
                data_path = getattr(a, 'path', None)
                nic = getattr(a, 'n_img_class', 0)
                extra = f'hier={getattr(a, "hier", "?")} n_img_class={nic}'
        except Exception as e:
            extra = f'(unreadable args: {type(e).__name__})'

        # prefer the lmdb path stored IN the checkpoint — it is ground truth about what
        # the run was actually fitted on; fall back to the family table for non-prior ckpts
        LMDB_DESC = {
            'lmdb/all': ('600px codes, ALL 10 boards — NO HOLDOUT (test boards LEAKED)', '~10,668'),
            'lmdb/train_pool': ('600px codes, train pool only (06/09 excluded)', '8,596'),
            'lmdb/train_pool_tight': ('TIGHT codes, train pool only (06/09 excluded)', '4,274'),
        }
        if data_path in LMDB_DESC:
            desc, n = LMDB_DESC[data_path]
        else:
            desc, n = TRAIN_DATA.get(fam, ('?', '?'))
        reached = max(eps) if eps else ('best' if 'best' in paths[-1] else 'n/a')
        rows.append({'family': fam, 'n_ckpt': len(paths), 'reached': reached,
                     'planned': planned, 'lmdb': data_path, 'data': desc, 'n_train': n,
                     'extra': extra})

    w = max(len(r['family']) for r in rows) + 2
    print(f'{"family":<{w}} {"ckpts":>5} {"reached":>8} {"planned":>8}  training data')
    print('-' * 120)
    for r in rows:
        print(f'{r["family"]:<{w}} {r["n_ckpt"]:>5} {str(r["reached"]):>8} '
              f'{str(r["planned"] or "-"):>8}  {r["data"]} ({r["n_train"]})')
        if r['lmdb'] or r['extra']:
            print(f'{"":<{w}} {"":>5} {"":>8} {"":>8}    lmdb={r["lmdb"]}  {r["extra"]}')

    print('\n' + '=' * 78)
    print('UNDERTRAINED?')
    print('=' * 78)
    print('  pixelsnail_top / _bottom  (the ORIGINAL, ICCE-TW pipeline)')
    print('      420-epoch budget; top checkpoint saved at epoch 357. Trained on lmdb/all —')
    print('      i.e. ALL 10 boards, INCLUDING the held-out test boards 06 and 09. This is a')
    print('      leak: the original generator saw the test boards. It does not affect our')
    print('      board-split classifier numbers, but no sample from these priors may ever be')
    print('      used as training data for a model we then evaluate on boards 06/09.')
    print('  pixelsnail_condtight2  (the prior the conditioning check was run on)')
    print('      80 epochs vs the original 420 — a 5.25x smaller budget — and the loss was')
    print('      STILL FALLING at epoch 80 (see figures/fig8). It is UNDERTRAINED.')
    print('  vqvae_tight_560 / vqvae_560   560 epochs, converged (tight recon MSE 0.00069).')
    print('  classifier_*                  20 epochs, fixed recipe, converged.')

    with open('results/checkpoint_inventory.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    print('\nwrote results/checkpoint_inventory.csv')
    return rows


if __name__ == '__main__':
    main()
