"""Post-hoc train vs held-out NLL for the conditional priors. Trains nothing.

TWO HONEST CAVEATS, stated up front because they change how the numbers read.

1. THERE IS NO HELD-OUT SLICE OF TRAIN-POOL CODES. The priors were trained on every code in
   lmdb/train_pool_tight (all 4,274). Any slice carved from it now is IN-SAMPLE, so it can
   only measure memorisation, never validation. That option does not exist post-hoc.

   The held-out set used here is therefore the TEST BOARDS (06/09): 518 tight crops that
   neither the priors nor the VQ-VAE (vqvae_tight_560, train-pool only, asserted) have ever
   seen. This is a HARDER test than a within-train holdout -- it asks the prior to generalise
   to unseen boards, not merely to unseen crops of seen boards. So the gap reported below is
   an UPPER bound on the memorisation gap: some of it is board shift, not memorisation.

2. THIS IS MEASUREMENT ONLY, NOT MODEL SELECTION. We never pick a checkpoint, tune, or gate
   on the held-out NLL -- doing so would leak the test boards into the experiment. The gate
   (gate_check.py) is untouched and does not consult these numbers.

Loss is exactly the training objective: teacher-forced CrossEntropy over the 512-way codebook,
top prior on top codes, bottom prior on bottom codes conditioned on GROUND-TRUTH top codes.
Reported in nats/code; chance for a 512-way codebook is ln(512) = 6.238.
"""
import sys, pathlib  # repo layout (README.md): src/, experiments/, analysis/ importable from anywhere
sys.path[1:1] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('src', 'experiments', 'analysis')]
import os
import json
import math
import argparse
import glob

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataset import LMDBDataset
from cond_check import load_prior

CHANCE = math.log(512)


@torch.no_grad()
def nll(model, loader, hier, device):
    """Mean teacher-forced NLL (nats/code) and code accuracy. Identical to the training loss."""
    crit = nn.CrossEntropyLoss(reduction='sum')
    tot_loss, tot_n, tot_correct = 0.0, 0, 0
    for top, bottom, label in loader:
        top, label = top.to(device), label.to(device)
        if hier == 'top':
            target = top
            out, _ = model(top, class_label=label)
        else:
            bottom = bottom.to(device)
            target = bottom
            out, _ = model(bottom, condition=top, class_label=label)
        tot_loss += crit(out, target).item()
        n = target.numel()
        tot_n += n
        tot_correct += (out.argmax(1) == target).sum().item()
    return tot_loss / tot_n, tot_correct / tot_n


def ckpts():
    """(absolute_epoch, top_path, bottom_path) for every checkpoint pair we have."""
    out = [(80, 'checkpoint/pixelsnail_condtight2_top_080.pt',
                'checkpoint/pixelsnail_condtight2_bottom_080.pt')]
    for t in sorted(glob.glob('checkpoint/pixelsnail_condtight3_top_*.pt')):
        L = int(t.rsplit('_', 1)[1].split('.')[0])
        b = f'checkpoint/pixelsnail_condtight3_bottom_{L:03d}.pt'
        if L % 40 == 0 and os.path.exists(b):        # every 40 epochs: 120,160,...,320
            out.append((80 + L, t, b))
    return [(e, t, b) for e, t, b in out if os.path.exists(t) and os.path.exists(b)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_lmdb', default='lmdb/train_pool_tight')
    ap.add_argument('--test_lmdb', default='lmdb/test_tight')
    ap.add_argument('--n_train', type=int, default=518)   # match the held-out set size exactly
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--out', default='results/prior_nll.json')
    args = ap.parse_args()
    device = 'cuda'

    tr_full = LMDBDataset(args.train_lmdb)
    te = LMDBDataset(args.test_lmdb)
    # fixed random slice of TRAIN codes, same size as the held-out set so NLLs are comparable
    idx = np.random.RandomState(0).choice(len(tr_full), min(args.n_train, len(tr_full)), replace=False)
    tr = Subset(tr_full, idx.tolist())
    print(f'train slice (IN-SAMPLE): {len(tr)} of {len(tr_full)}   '
          f'held-out (test boards 06/09): {len(te)}')
    print(f'chance NLL for a 512-way codebook = ln(512) = {CHANCE:.3f} nats/code\n')

    tl = DataLoader(tr, batch_size=args.batch, num_workers=4)
    vl = DataLoader(te, batch_size=args.batch, num_workers=4)

    rows = []
    print(f'{"epoch":>6} {"hier":>7} {"train NLL":>10} {"heldout NLL":>12} {"gap":>8} '
          f'{"train acc":>10} {"heldout acc":>12}')
    print('-' * 74)
    for ep, tp, bp in ckpts():
        rec = {'epoch': ep}
        for hier, path in (('top', tp), ('bottom', bp)):
            m = load_prior(path, device)
            ltr, atr = nll(m, tl, hier, device)
            lte, ate = nll(m, vl, hier, device)
            del m
            torch.cuda.empty_cache()
            rec[hier] = {'train_nll': ltr, 'heldout_nll': lte, 'gap': lte - ltr,
                         'train_acc': atr, 'heldout_acc': ate}
            print(f'{ep:>6} {hier:>7} {ltr:>10.4f} {lte:>12.4f} {lte-ltr:>+8.4f} '
                  f'{atr:>10.4f} {ate:>12.4f}', flush=True)
        rows.append(rec)

    with open(args.out, 'w') as fh:
        json.dump(rows, fh, indent=2)

    # ---- the two questions ----
    print('\n' + '=' * 74)
    for hier in ('top', 'bottom'):
        v = [(r['epoch'], r[hier]) for r in rows]
        if len(v) < 2:
            continue
        e0, a = v[-2]
        e1, b = v[-1]
        d_tr = b['train_nll'] - a['train_nll']
        d_te = b['heldout_nll'] - a['heldout_nll']
        print(f'\n{hier.upper()} prior')
        print(f'  CONVERGED AT {e1}?  held-out NLL {a["heldout_nll"]:.4f} (ep {e0}) -> '
              f'{b["heldout_nll"]:.4f} (ep {e1})   change {d_te:+.4f}')
        if d_te < -0.002:
            print('     -> held-out NLL STILL FALLING: not converged, more epochs would help')
        elif d_te > 0.002:
            print('     -> held-out NLL RISING while train falls: OVERFITTING past this point')
        else:
            print('     -> held-out NLL FLAT: converged')
        print(f'  MEMORISED THE TRAIN BOARDS?  gap = {b["gap"]:+.4f} nats/code '
              f'(train {b["train_nll"]:.4f} vs held-out {b["heldout_nll"]:.4f})')
        print(f'     train acc {b["train_acc"]:.3f} vs held-out acc {b["heldout_acc"]:.3f}')
        gap0 = rows[0][hier]['gap']
        print(f'     gap at epoch {rows[0]["epoch"]} was {gap0:+.4f} -> now {b["gap"]:+.4f} '
              f'({"WIDENING = memorising" if b["gap"] > gap0 + 0.01 else "stable"})')
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
