"""Is a high nearest-neighbour similarity actually evidence of copying? Calibrate it.

A cosine of 0.99 to the nearest training crop sounds damning, but PCB crops all look alike --
green solder mask, copper traces, silver pads. Two unrelated boards may already sit at 0.95 in
an ImageNet feature space. Without a null, the number is meaningless (the same mistake the
saturated defect detector made).

So we compute the SAME statistic for images that are provably NOT copies:
  NULL      real TEST-board crops -> NN in the training set. These are genuine photographs the
            generator never saw. Whatever similarity they score is the "not a copy" baseline.
  NULL-2    real TRAIN crops -> NN in the training set, EXCLUDING the image itself. Same
            distribution, same boards, still not copies of their neighbour.
  CEILING   VQ-VAE reconstructions of training crops -> NN in the training set. These ARE the
            training image, round-tripped. This is what a perfect copy scores.
  TEST      the generated samples.
Only if generated sits near the CEILING and clearly above the NULL is "memorisation" earned.
"""
import glob
import json
import os
import random

import numpy as np
import torch
from PIL import Image

from build_review import load_rows
from build_generation_review import embed, pick_device

SEED = 0


def nn_sim(query_paths, bank_paths, device, exclude_self=False):
    q = embed(query_paths, device)
    b = embed(bank_paths, device)
    sim = q @ b.T
    if exclude_self:
        for i, p in enumerate(query_paths):
            for j, r in enumerate(bank_paths):
                if os.path.basename(p) == os.path.basename(r):
                    sim[i, j] = -1
    return sim.max(1).values.numpy()


def main():
    device = pick_device()
    rng = random.Random(SEED)
    print(f'device: {device}\n')

    # ---------- TIGHT domain ----------
    tt = load_rows('results/manifest_tight_perbbox.csv')
    tight_train = [r['crop_path'] for r in tt if r['split'] == 'train']
    tight_test = [r['crop_path'] for r in tt if r['split'] == 'test']
    gen_cond = sorted(glob.glob('genreview_assets/cond/*.png'))
    recon_tr = [p for p in (os.path.join('recon_pool', os.path.basename(x))
                            for x in rng.sample(tight_train, 120)) if os.path.exists(p)]

    print('TIGHT domain  (bank = the 4,274 tight crops the prior trained on)')
    rows = []
    for name, q, ex in [
        ('NULL    real TEST-board crops (never seen)', rng.sample(tight_test, 120), False),
        ('NULL-2  real TRAIN crops (self excluded)', rng.sample(tight_train, 120), True),
        ('CEILING VQ-VAE recon of TRAIN crops (a real copy)', recon_tr, False),
        ('TEST    GENERATED conditional (ep320)', gen_cond, False),
    ]:
        if not q:
            continue
        v = nn_sim(q, tight_train, device, ex)
        rows.append({'domain': 'tight', 'set': name, 'n': len(q), 'mean': float(v.mean()),
                     'median': float(np.median(v)), 'p95': float(np.percentile(v, 95))})
        print(f'  {name:<50} n={len(q):3d}  mean {v.mean():.3f}  median {np.median(v):.3f}')

    # ---------- 600px domain ----------
    all600 = [r['crop_path'] for r in load_rows('results/manifest.csv')]
    # the original priors trained on lmdb/all = every crop, so there is no unseen real crop in
    # this domain. Use the TEST-board crops anyway: they were in the prior's training set, which
    # only makes the NULL *more* favourable to a memorisation verdict -- and it still holds.
    recon600 = sorted(glob.glob('recon_600/*.png'))
    print('\n600px domain  (bank = the 10,668 crops the ORIGINAL prior trained on)')
    for name, q in [
        ('NULL-2  real crops (self excluded)', rng.sample(all600, 120)),
        ('CEILING VQ-VAE recon of real crops (a real copy)', recon600[:120]),
        ('TEST    GENERATED demo_pool', rng.sample(sorted(glob.glob('demo_pool/*.png')), 120)),
        ('TEST    GENERATED my sample_uncond', sorted(glob.glob('synth_uncond/u*.png'))[:120]),
    ]:
        if not q:
            continue
        ex = name.startswith('NULL-2')
        v = nn_sim(q, all600, device, ex)
        rows.append({'domain': '600px', 'set': name, 'n': len(q), 'mean': float(v.mean()),
                     'median': float(np.median(v)), 'p95': float(np.percentile(v, 95))})
        print(f'  {name:<50} n={len(q):3d}  mean {v.mean():.3f}  median {np.median(v):.3f}')

    json.dump(rows, open('results/nn_baseline.json', 'w'), indent=2)

    print('\n' + '=' * 78)
    print('VERDICT — is the generator copying?')
    print('=' * 78)
    for dom in ('tight', '600px'):
        d = {r['set'][:7].strip(): r for r in rows if r['domain'] == dom}
        null = d.get('NULL') or d.get('NULL-2')
        ceil, test = d.get('CEILING'), None
        tests = [r for r in rows if r['domain'] == dom and r['set'].startswith('TEST')]
        if not (null and ceil and tests):
            continue
        print(f'\n{dom}:  null {null["mean"]:.3f}   copy-ceiling {ceil["mean"]:.3f}')
        span = ceil['mean'] - null['mean']
        for t in tests:
            frac = (t['mean'] - null['mean']) / span if span > 1e-6 else float('nan')
            print(f'   {t["set"][8:]:<34} {t["mean"]:.3f}   -> {100*frac:5.1f}% of the way '
                  f'from "novel" to "a literal copy"')
    print('\nwrote results/nn_baseline.json')


if __name__ == '__main__':
    main()
