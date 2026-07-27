"""Copy detection in PIXEL space, because the ImageNet feature space is degenerate here.

Every 600px whole-board crop embeds to nearly the same ImageNet vector -- green solder mask,
copper traces -- so two COMPLETELY DIFFERENT boards score cosine 1.000 and the metric has no
dynamic range at all. Any "memorisation" verdict built on it is an artefact. (It works fine on
tight crops, which are visually varied: null 0.888 vs copy-ceiling 0.983.)

A copy is a copy in pixels, so we measure there: 64x64 RGB, per-image mean-centred, cosine.
This has real dynamic range on board images and is the standard way to catch a generative model
regurgitating its training set.

Always with a null and a ceiling:
  NULL     a real crop vs the nearest OTHER base crop (its own rotations/flips excluded).
           Two genuinely different boards. This is "not a copy".
  CEILING  a VQ-VAE reconstruction vs the training set. It IS the training image, round-tripped
           -- the same decoder any generated sample passes through. This is "a literal copy".
  TEST     the generated samples.
Generated near CEILING => regurgitation. Generated near NULL => genuinely novel boards.
"""
import glob
import json
import os
import random

import numpy as np
import torch
from PIL import Image

from pcb_utils import parse_filename
from build_review import load_rows

R = 64
SEED = 0


def feats(paths, device):
    out = []
    for i in range(0, len(paths), 256):
        arr = []
        for p in paths[i:i + 256]:
            im = Image.open(p).convert('RGB').resize((R, R), Image.LANCZOS)
            arr.append(np.asarray(im, dtype=np.float32).ravel() / 255.0)
        x = torch.from_numpy(np.stack(arr)).to(device)
        x = x - x.mean(1, keepdim=True)                 # kill the global green DC term
        out.append(torch.nn.functional.normalize(x, dim=1).cpu())
    return torch.cat(out)


def key(p):
    q = parse_filename(os.path.basename(p))
    return None if q is None else (q['board'], q['cls'], q['mm'], q['k'])


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = random.Random(SEED)
    bank = [r['crop_path'] for r in load_rows('results/manifest.csv')]
    bk = [key(p) for p in bank]
    B = feats(bank, device).to(device)
    print(f'pixel-space copy detection ({R}x{R}, mean-centred). bank = {len(bank)} crops\n')

    def nn(q, drop_same=False):
        Q = feats(q, device).to(device)
        sim = Q @ B.T
        if drop_same:
            for i, p in enumerate(q):
                k = key(p)
                if k is None:
                    continue
                m = torch.tensor([x == k for x in bk], device=device)
                sim[i, m] = -1
        return sim.max(1).values.cpu().numpy()

    rows = [
        ('NULL    real crop -> nearest OTHER board (own rotations excluded)',
         nn(rng.sample(bank, 150), drop_same=True)),
        ('CEILING VQ-VAE recon of a real crop (a literal copy)',
         nn(sorted(glob.glob('recon_600/*.png'))[:150])),
        ('TEST    GENERATED demo_pool',
         nn(rng.sample(sorted(glob.glob('demo_pool/*.png')), 150))),
        ('TEST    GENERATED my sample_uncond',
         nn(sorted(glob.glob('synth_uncond/u*.png')))),
    ]
    for n, v in rows:
        print(f'  {n:<62} n={len(v):3d}  mean {v.mean():.3f}  median {np.median(v):.3f}  '
              f'max {v.max():.3f}')

    null, ceil = rows[0][1].mean(), rows[1][1].mean()
    span = ceil - null
    print('\n' + '=' * 84)
    print(f'  null (a genuinely different board) = {null:.3f}')
    print(f'  ceiling (the same image, round-tripped through the decoder) = {ceil:.3f}')
    print(f'  dynamic range = {span:.3f}' + ('   <-- usable' if span > 0.05 else
                                             '   <-- TOO SMALL, metric unusable'))
    for n, v in rows[2:]:
        f = (v.mean() - null) / span if abs(span) > 1e-6 else float('nan')
        verdict = ('REGURGITATING the training set' if f > 0.7 else
                   'genuinely novel — NOT copying' if f < 0.3 else 'partial overlap')
        print(f'  {n[8:]:<34} {v.mean():.3f}  -> {100*f:5.1f}% toward a literal copy  [{verdict}]')
    json.dump([{'set': n, 'mean': float(v.mean()), 'max': float(v.max()), 'n': len(v)}
               for n, v in rows], open('results/nn_pixel.json', 'w'), indent=2)
    print('\nwrote results/nn_pixel.json')


if __name__ == '__main__':
    main()
