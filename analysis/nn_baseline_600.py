"""Fix the 600px null: exclude a crop's own augmented variants from its neighbour bank.

The bank (lmdb/all, 10,668 crops) contains 4 variants of every base crop -- plain, l, rot90,
rot270. So a real crop's nearest neighbour is its OWN rotation, and the null came out at 1.000,
which is not a null at all, it is self-matching. Excluding every variant that shares the same
(board, class, mm, k) gives the real "two different crops of the same kind of board" baseline,
which is the number a generated image has to beat to count as a copy.
"""
import sys, pathlib  # repo layout (README.md): src/, experiments/, analysis/ importable from anywhere
sys.path[1:1] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('src', 'experiments', 'analysis')]
import glob
import json
import os
import random

import numpy as np
import torch

from pcb_utils import parse_filename
from build_review import load_rows
from build_generation_review import embed, pick_device

SEED = 0


def key(p):
    q = parse_filename(os.path.basename(p))
    return None if q is None else (q['board'], q['cls'], q['mm'], q['k'])


def main():
    device = pick_device()
    rng = random.Random(SEED)
    bank = [r['crop_path'] for r in load_rows('results/manifest.csv')]
    bank_keys = [key(p) for p in bank]
    B = embed(bank, device)
    print(f'bank: {len(bank)} crops (all variants)\n')

    def nn(qpaths, drop_same_base=False):
        Q = embed(qpaths, device)
        sim = Q @ B.T
        if drop_same_base:
            for i, p in enumerate(qpaths):
                k = key(p)
                if k is None:
                    continue
                mask = torch.tensor([bk == k for bk in bank_keys])
                sim[i, mask] = -1          # drop the crop AND all its rotations/flips
        return sim.max(1).values.numpy()

    rows = []
    reals = rng.sample(bank, 150)
    v_null = nn(reals, drop_same_base=True)
    rows.append(('NULL    real crop -> nearest OTHER base crop (not its own rotation)', v_null))

    recon = sorted(glob.glob('recon_600/*.png'))[:150]
    if recon:
        rows.append(('CEILING VQ-VAE recon of a real crop (a literal copy)', nn(recon)))

    rows.append(('TEST    GENERATED demo_pool',
                 nn(rng.sample(sorted(glob.glob('demo_pool/*.png')), 150))))
    rows.append(('TEST    GENERATED my sample_uncond',
                 nn(sorted(glob.glob('synth_uncond/u*.png')))))

    print('600px domain, self-variants EXCLUDED from the bank')
    for name, v in rows:
        print(f'  {name:<58} n={len(v):3d}  mean {v.mean():.3f}  median {np.median(v):.3f}')

    null = rows[0][1].mean()
    ceil = rows[1][1].mean() if len(rows) > 2 else None
    print('\n' + '=' * 80)
    if ceil:
        span = ceil - null
        print(f'null (a genuinely different board) = {null:.3f}')
        print(f'copy-ceiling (the same image, round-tripped) = {ceil:.3f}')
        for name, v in rows[2:]:
            f = (v.mean() - null) / span if abs(span) > 1e-6 else float('nan')
            verdict = ('COPYING' if f > 0.7 else
                       'not copying — indistinguishable from a novel board' if f < 0.3 else
                       'partial')
            print(f'  {name[8:]:<34} {v.mean():.3f}  -> {100*f:5.1f}% toward a literal copy  [{verdict}]')
    json.dump([{'set': n, 'mean': float(v.mean()), 'median': float(np.median(v)), 'n': len(v)}
               for n, v in rows], open('results/nn_baseline_600.json', 'w'), indent=2)
    print('\nwrote results/nn_baseline_600.json')


if __name__ == '__main__':
    main()
