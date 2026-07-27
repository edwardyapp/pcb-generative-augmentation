"""Score every pool with the in-domain 600px detector, using LOGIT MARGINS.

Why not softmax. The detector is confident: softmax P(defect) saturates at 1.000 on clean
windows and defect windows alike, so max-over-225-windows is 1.000 for every image and the
statistic carries zero information. The LOGIT MARGIN (logit_defect - logit_clean) does not
saturate, so it can actually rank images.

Statistic: the mean of the TOP-3 window margins. A real defect lights up a small CLUSTER of
overlapping windows (stride 14 < window 55), so top-3 is robust; a lone spurious window is not
enough to carry it.

Calibration, all on HELD-OUT boards:
  NULL    = the same statistic computed over only the provably-CLEAN windows of real crops
            (the bboxes tell us where the defect is NOT). Threshold = 95th pct => 5% FPR.
  CEILING = VQ-VAE RECONSTRUCTIONS of those same real crops. Same decoder blur and codebook
            quantisation as a generated image, but a REAL defect inside. This is the number a
            generator would have to match. It is the control that makes the comparison fair:
            any shortfall below it CANNOT be blamed on decoder blur.
"""
import os
import glob
import json
import random
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn
from torchvision import models
from PIL import Image

from pcb_utils import parse_filename, load_splits
from detector_600 import WIN, STRIDE, S, SCALE, boxes_256, make_recons, MEAN, STD

TOPK = 3
SEED = 0


@torch.no_grad()
def margins(path, m, device):
    im = Image.open(path).convert('RGB').resize((S, S))
    x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255)[None].to(device)
    pat = x.unfold(2, WIN, STRIDE).unfold(3, WIN, STRIDE)
    nh, nw = pat.shape[2], pat.shape[3]
    pat = pat.permute(0, 2, 3, 1, 4, 5).reshape(nh * nw, 3, WIN, WIN)
    pat = torch.nn.functional.interpolate(pat, (224, 224), mode='bilinear', align_corners=False)
    pat = (pat - MEAN.to(device)) / STD.to(device)
    lg = []
    for i in range(0, pat.shape[0], 512):
        o = m(pat[i:i + 512])
        lg.append(o[:, 1] - o[:, 0])                  # logit margin, no saturation
    return torch.cat(lg).cpu().numpy().reshape(nh, nw)


def topk_mean(v):
    v = np.sort(np.asarray(v).ravel())[::-1]
    return float(v[:TOPK].mean()) if v.size else float('nan')


def main():
    device = 'cuda'
    m = models.resnet18()
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load('checkpoint/defect_detector_600.pt', map_location='cpu'))
    m = m.to(device).eval()
    rng = random.Random(SEED)

    test_boards = set(load_splits()['test_boards'])
    real_te = [p for p in sorted(glob.glob('VOC_PCB/JPEGImages/*.jpg'))
               if (q := parse_filename(os.path.basename(p))) and q['variant'] == 'plain'
               and q['board'] in test_boards]
    real_te = rng.sample(real_te, min(120, len(real_te)))

    # ---- NULL: same statistic, but only over windows that provably contain NO defect ----
    null, sig = [], []
    for p in real_te:
        bs = boxes_256(os.path.basename(p)[:-4])
        mg = margins(p, m, device)
        clean, hit = [], []
        for i in range(mg.shape[0]):
            for j in range(mg.shape[1]):
                x0, y0 = j * STRIDE, i * STRIDE
                x1, y1 = x0 + WIN, y0 + WIN
                near = any(not (x1 + 8 <= bx0 or x0 - 8 >= bx1 or
                                y1 + 8 <= by0 or y0 - 8 >= by1) for bx0, by0, bx1, by1 in bs)
                (hit if near else clean).append(mg[i, j])
        null.append(topk_mean(clean))
        sig.append(topk_mean(hit))
    null, sig = np.array(null), np.array(sig)
    T = float(np.percentile(null, 95))
    print('=' * 88)
    print('CALIBRATION (held-out boards 06/09; statistic = mean of top-3 window logit margins)')
    print('=' * 88)
    print(f'  NULL   (provably CLEAN windows of real crops): median {np.median(null):+.2f}')
    print(f'  SIGNAL (windows on the real defect)          : median {np.median(sig):+.2f}')
    print(f'  threshold T = {T:+.2f}  (95th pct of NULL => 5% false-positive rate)')
    print(f'  sensitivity at T on real defects: {100*(sig > T).mean():.1f}%')

    recon = make_recons(real_te, device)
    pools = [
        ('REAL 600px crops (photo, has a defect)', real_te),
        ('VQ-VAE RECON of them (decoder blur + REAL defect) <- CEILING', recon),
        ('GENERATED - demo_pool (your ICCE-TW demo)', rng.sample(sorted(glob.glob('demo_pool/*.png')), 120)),
        ('GENERATED - my sample_uncond (same ckpts, same sampler)', sorted(glob.glob('synth_uncond/u*.png'))),
    ]
    print('\n' + '=' * 88)
    print('DEFECT PRESENT?   (calibrated to 5% false-positive rate on real clean board)')
    print('=' * 88)
    out = []
    for name, ps in pools:
        v = np.array([topk_mean(margins(p, m, device)) for p in ps])
        frac = float((v > T).mean())
        out.append({'pool': name, 'n': len(ps), 'frac': frac, 'median_stat': float(np.median(v))})
        print(f'  {name:<60} n={len(ps):3d}  {100*frac:5.1f}%  (median stat {np.median(v):+.2f})')

    json.dump({'threshold': T, 'sensitivity': float((sig > T).mean()), 'pools': out},
              open('results/defect_presence_600.json', 'w'), indent=2)

    d = out[2]['frac']; mine = out[3]['frac']; ceil = out[1]['frac']
    print('\n' + '=' * 88)
    print(f'  demo_pool {100*d:.1f}%  vs  my samples {100*mine:.1f}%   '
          f'-> {abs(d-mine)*100:.1f} pp apart '
          f'({"SAME distribution; my sampler is fine" if abs(d-mine) < 0.10 else "DIFFERENT - investigate"})')
    print(f'  ceiling (real defect + same decoder blur) = {100*ceil:.1f}%')
    print('\nwrote results/defect_presence_600.json')


if __name__ == '__main__':
    main()
