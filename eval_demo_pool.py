"""Settle it: does the ORIGINAL unconditional pipeline draw defects, or not?

demo_pool (the ICCE-TW demo) and my sample_uncond samples come from the SAME checkpoints,
the SAME sampler, the SAME temperature and the SAME latent shapes -- generate_pool.py only
LANCZOS-upscales the 256px render to 512 for display. So they must agree. If they do not,
my sampler is broken and I want to know.

Every pool is scored with the same held-out-validated defect detector (95.0% accurate on
boards 06/09), and calibrated against REAL crops scored by the identical procedure:
  * 600px domain (whole-board views): the detector expects a defect to fill the frame, so it
    is SLID across the image (55px windows ~ 128px native, stride 14). A defect is present if
    any window fires.
  * tight domain: applied directly.
The real-crop rows are the calibration. A generated number means nothing without them.
"""
import glob
import json
import random

import numpy as np
import torch

from build_review import load_rows
from build_generation_review import defect_fraction, defect_fraction_600, pick_device

N = 120     # sample size per pool (sliding-window is ~225 forward passes per image)
SEED = 0


def main():
    device = pick_device()
    print(f'device: {device}\n')
    rng = random.Random(SEED)

    def take(paths, n=N):
        return rng.sample(sorted(paths), min(n, len(paths)))

    real600 = [r['crop_path'] for r in load_rows('results/manifest.csv')]
    real_tight = [r['crop_path'] for r in load_rows('results/manifest_tight.csv')]
    demo = glob.glob('demo_pool/*.png')
    mine = glob.glob('synth_uncond/u*.png')
    cond = glob.glob('genreview_assets/cond/*.png') or glob.glob('recon_pool/*.png')[:0]

    rows = []

    # ---- 600px domain: sliding window ----
    print('600px domain (sliding window) ------------------------------------')
    for name, paths, kind in [
        ('REAL 600px crops (every one CONTAINS a defect)', real600, 'real'),
        ('GENERATED — demo_pool (ICCE-TW demo, 512px)', demo, 'gen'),
        ('GENERATED — my sample_uncond (256px, same ckpts)', mine, 'gen'),
    ]:
        p = take(paths)
        frac, scores = defect_fraction_600(p, device)
        rows.append({'pool': name, 'n': len(p), 'frac': frac,
                     'mean_score': float(np.mean(scores)), 'kind': kind, 'domain': '600px'})
        print(f'  {name:<52} n={len(p):3d}  defect present: {100*frac:5.1f}%  '
              f'(mean max-score {np.mean(scores):.3f})')

    # ---- tight domain: direct ----
    print('\ntight domain (direct) --------------------------------------------')
    for name, paths, kind in [
        ('REAL tight crops (every one CONTAINS a defect)', real_tight, 'real'),
        ('GENERATED — conditional tight generator', cond, 'gen'),
    ]:
        if not paths:
            print(f'  {name:<52} (not available yet)')
            continue
        p = take(paths)
        frac, scores = defect_fraction(p, device)
        rows.append({'pool': name, 'n': len(p), 'frac': frac,
                     'mean_score': float(np.mean(scores)), 'kind': kind, 'domain': 'tight'})
        print(f'  {name:<52} n={len(p):3d}  defect present: {100*frac:5.1f}%  '
              f'(mean score {np.mean(scores):.3f})')

    json.dump(rows, open('results/defect_presence.json', 'w'), indent=2)

    # ---- the decisive comparison ----
    d = next((r for r in rows if 'demo_pool' in r['pool']), None)
    m = next((r for r in rows if 'sample_uncond' in r['pool']), None)
    r6 = next((r for r in rows if r['kind'] == 'real' and r['domain'] == '600px'), None)
    if d and m:
        print('\n' + '=' * 70)
        print(f'demo_pool          {100*d["frac"]:5.1f}% defect present')
        print(f'my sample_uncond   {100*m["frac"]:5.1f}% defect present')
        print(f'REAL 600px crops   {100*r6["frac"]:5.1f}% defect present   <- calibration')
        gap = abs(d['frac'] - m['frac'])
        print(f'\ndemo_pool vs mine: {100*gap:.1f} pp apart')
        if gap < 0.10:
            print('  => SAME distribution, as the code diff predicts. My sampler is not broken.')
        else:
            print('  => THEY DIFFER. Something in the sampling path is not equivalent — investigate.')
    print('\nwrote results/defect_presence.json')


if __name__ == '__main__':
    main()
