"""Calibrate the sliding-window defect test properly, because the naive version is saturated.

THE BUG IN THE NAIVE TEST. The detector's per-window false-positive rate is ~5.7%. Sliding it
over an image gives ~225 windows. Scoring an image positive if ANY window fires means
    P(false positive) = 1 - 0.943^225 ~ 1.0
so the test says "defect present" for essentially any image, including a blank board. That is
exactly why it returned 100% on real crops, 100% on my samples and 96.7% on demo_pool -- those
numbers carry no information at all.

THE FIX. Use the bounding boxes to split the windows of a REAL 600px crop into:
   * CLEAN windows  - miss every bbox by a margin -> a defect is provably NOT there
   * DEFECT windows - overlap a bbox             -> a defect IS there
The image-level score is max P(defect) over a set of windows. Then:
   * max over CLEAN windows, across real crops  -> the NULL distribution
   * pick threshold T at the 95th percentile of that null -> a 5% image-level false-positive rate
   * max over DEFECT windows -> sensitivity at T
   * apply the SAME T to the generated pools (where we do not know where a defect would be, so
     all windows are used -- exactly as for the null, which also uses a whole-image max)
Now "defect present" means something, and it is calibrated on real data.
"""
import glob
import json
import random
import xml.etree.ElementTree as ET
import os

import numpy as np
import torch
from torch import nn
from torchvision import models
from PIL import Image

from pcb_utils import parse_filename
from build_generation_review import pick_device

WIN, STRIDE, S = 55, 14, 256           # 55px in a 256 decode ~ 128px native in a 600px frame
SCALE = S / 600.0
N = 150
SEED = 0


def load_det(device):
    m = models.resnet18()
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load('checkpoint/defect_detector.pt', map_location='cpu'))
    return m.to(device).eval()


@torch.no_grad()
def window_scores(path, m, device):
    """P(defect) for every sliding window. Returns (scores, xs, ys)."""
    im = Image.open(path).convert('RGB').resize((S, S))
    x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255)[None].to(device)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    pat = x.unfold(2, WIN, STRIDE).unfold(3, WIN, STRIDE)
    nh, nw = pat.shape[2], pat.shape[3]
    pat = pat.permute(0, 2, 3, 1, 4, 5).reshape(nh * nw, 3, WIN, WIN)
    pat = torch.nn.functional.interpolate(pat, (224, 224), mode='bilinear', align_corners=False)
    pat = (pat - mean) / std
    sc = []
    for i in range(0, pat.shape[0], 512):
        sc.append(m(pat[i:i + 512]).softmax(1)[:, 1])
    sc = torch.cat(sc).cpu().numpy().reshape(nh, nw)
    ys = np.arange(nh) * STRIDE
    xs = np.arange(nw) * STRIDE
    return sc, xs, ys


def boxes_256(base):
    p = f'VOC_PCB/Annotations/{base}.xml'
    if not os.path.exists(p):
        return []
    out = []
    for o in ET.parse(p).getroot().findall('object'):
        b = o.find('bndbox')
        out.append(tuple(int(b.find(k).text) * SCALE
                         for k in ('xmin', 'ymin', 'xmax', 'ymax')))
    return out


def split_windows(sc, xs, ys, bs, margin=6):
    """max over windows that provably MISS every bbox, and max over windows that HIT one."""
    clean, hit = [], []
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            x0, y0, x1, y1 = x, y, x + WIN, y + WIN
            overlaps = any(not (x1 <= bx0 or x0 >= bx1 or y1 <= by0 or y0 >= by1)
                           for bx0, by0, bx1, by1 in bs)
            near = any(not (x1 + margin <= bx0 or x0 - margin >= bx1 or
                            y1 + margin <= by0 or y0 - margin >= by1)
                       for bx0, by0, bx1, by1 in bs)
            if overlaps:
                hit.append(sc[i, j])
            elif not near:
                clean.append(sc[i, j])
    return (max(clean) if clean else np.nan), (max(hit) if hit else np.nan)


def main():
    device = pick_device()
    m = load_det(device)
    rng = random.Random(SEED)

    # real 600px crops with annotations (plain variant only -> XML exists)
    reals = [p for p in sorted(glob.glob('VOC_PCB/JPEGImages/*.jpg'))
             if (q := parse_filename(os.path.basename(p))) and q['variant'] == 'plain']
    reals = rng.sample(reals, min(N, len(reals)))

    null, sens = [], []
    for p in reals:
        base = os.path.basename(p)[:-4]
        sc, xs, ys = window_scores(p, m, device)
        c, h = split_windows(sc, xs, ys, boxes_256(base))
        if not np.isnan(c):
            null.append(c)
        if not np.isnan(h):
            sens.append(h)
    null, sens = np.array(null), np.array(sens)

    T = float(np.percentile(null, 95))          # 5% image-level FPR on provably-clean regions
    print('=' * 78)
    print('CALIBRATION on REAL 600px crops (the bboxes tell us where a defect is and is not)')
    print('=' * 78)
    print(f'  images: {len(reals)}   windows/image ~ {len(xs)*len(ys)}')
    print(f'  NULL   max P(defect) over provably-CLEAN windows: '
          f'median {np.median(null):.3f}  95th pct {T:.3f}')
    print(f'  SIGNAL max P(defect) over DEFECT windows:         '
          f'median {np.median(sens):.3f}')
    print(f'\n  naive threshold 0.5 -> false-positive rate on CLEAN regions = '
          f'{100*(null > 0.5).mean():.1f}%   <-- the naive test is SATURATED, it is useless')
    print(f'  calibrated threshold T = {T:.3f} -> 5.0% FPR by construction, '
          f'sensitivity {100*(sens > T).mean():.1f}% on real defects')

    # ---- apply the calibrated threshold to every pool ----
    print('\n' + '=' * 78)
    print(f'DEFECT PRESENT?  (whole-image max window score > T = {T:.3f}; 5% FPR, '
          f'{100*(sens>T).mean():.0f}% sensitivity)')
    print('=' * 78)
    pools = [('REAL 600px crops (all contain a defect)', reals),
             ('GENERATED - demo_pool (ICCE-TW demo)', rng.sample(sorted(glob.glob('demo_pool/*.png')), N)),
             ('GENERATED - my sample_uncond (same ckpts)', sorted(glob.glob('synth_uncond/u*.png')))]
    out = []
    for name, ps in pools:
        mx = []
        for p in ps:
            sc, xs, ys = window_scores(p, m, device)
            mx.append(float(sc.max()))
        mx = np.array(mx)
        frac = float((mx > T).mean())
        out.append({'pool': name, 'n': len(ps), 'frac_calibrated': frac,
                    'median_max_score': float(np.median(mx)), 'threshold': T})
        print(f'  {name:<44} n={len(ps):3d}   {100*frac:5.1f}%   '
              f'(median max score {np.median(mx):.3f})')

    json.dump({'threshold': T, 'null_95': T, 'sensitivity': float((sens > T).mean()),
               'naive_fpr_at_0.5': float((null > 0.5).mean()), 'pools': out},
              open('results/defect_presence_calibrated.json', 'w'), indent=2)
    print('\nwrote results/defect_presence_calibrated.json')


if __name__ == '__main__':
    main()
