"""An IN-DOMAIN defect detector for the 600px whole-board scale, and the decisive control.

Why the previous attempt failed. The tight detector was trained on sharp 128px-native crops.
Sliding it over a 256px decode of a 600px frame feeds it 55px patches of already-blurred,
VQ-VAE-decoded content, upsampled 4x -- wildly out of domain. It fired on 81.3% of provably
CLEAN regions. Any number built on it is noise.

This detector is trained on EXACTLY the windows it will be asked to score:
  * take real 600px crops, resize to 256 (the size the generator decodes to)
  * slide 55px windows (the same WIN/STRIDE the scorer uses)
  * POSITIVE = the window contains a bbox CENTRE (the defect is really in there)
  * NEGATIVE = the window misses every bbox by a margin (a defect is provably NOT in there)
  * train on the train boards, validate on the HELD-OUT boards
So the detector sees the same blur, the same scale and the same upsampling at train and test.

THE CONTROL THAT MATTERS. Generated crops are VQ-VAE decodes: soft, smoothed, codebook-quantised.
Real crops are photographs. A detector could separate those two on texture alone and tell us
nothing about defects. So we also score VQ-VAE RECONSTRUCTIONS of real 600px crops -- identical
decoder artefacts, but they provably CONTAIN a defect. That is the ceiling a generator is
measured against:
      real photo  >=  VQ-VAE recon of real  >>  generated ?
If generated scores like the recon, the generator draws defects. If it scores far below, it
does not -- and the gap cannot be blamed on decoder blur, because the recon has the same blur.
"""
import os
import glob
import json
import random
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from pcb_utils import parse_filename, load_splits
from vqvae import VQVAE

WIN, STRIDE, S = 55, 14, 256
SCALE = S / 600.0
CKPT = 'checkpoint/defect_detector_600.pt'
SEED = 0
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def boxes_256(base):
    p = f'VOC_PCB/Annotations/{base}.xml'
    if not os.path.exists(p):
        return []
    out = []
    for o in ET.parse(p).getroot().findall('object'):
        b = o.find('bndbox')
        out.append(tuple(int(b.find(k).text) * SCALE for k in ('xmin', 'ymin', 'xmax', 'ymax')))
    return out


class WinDS(Dataset):
    """Windows cut on the fly from real 600px crops resized to 256."""

    def __init__(self, items, aug):
        self.items, self.aug = items, aug

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, x, y, lab = self.items[i]
        im = Image.open(path).convert('RGB').resize((S, S))
        w = im.crop((x, y, x + WIN, y + WIN)).resize((224, 224), Image.BILINEAR)
        if self.aug and random.random() < 0.5:
            w = w.transpose(Image.FLIP_LEFT_RIGHT)
        if self.aug and random.random() < 0.5:
            w = w.transpose(Image.FLIP_TOP_BOTTOM)
        t = transforms.functional.to_tensor(w)
        t = transforms.functional.normalize(t, MEAN.flatten().tolist(), STD.flatten().tolist())
        return t, lab


def build_windows():
    """POS = window contains a bbox centre. NEG = window clear of every bbox by a margin."""
    test_boards = set(load_splits()['test_boards'])
    tr, te = [], []
    rng = random.Random(SEED)
    for jp in sorted(glob.glob('VOC_PCB/JPEGImages/*.jpg')):
        base = os.path.basename(jp)[:-4]
        q = parse_filename(base + '.jpg')
        if q is None or q['variant'] != 'plain':
            continue
        bs = boxes_256(base)
        if not bs:
            continue
        split = te if q['board'] in test_boards else tr
        pos, neg = [], []
        for y in range(0, S - WIN + 1, STRIDE):
            for x in range(0, S - WIN + 1, STRIDE):
                x0, y0, x1, y1 = x, y, x + WIN, y + WIN
                has_centre = any(x0 <= (bx0 + bx1) / 2 <= x1 and y0 <= (by0 + by1) / 2 <= y1
                                 for bx0, by0, bx1, by1 in bs)
                near = any(not (x1 + 8 <= bx0 or x0 - 8 >= bx1 or
                                y1 + 8 <= by0 or y0 - 8 >= by1) for bx0, by0, bx1, by1 in bs)
                if has_centre:
                    pos.append((jp, x, y, 1))
                elif not near:
                    neg.append((jp, x, y, 0))
        split += pos
        split += rng.sample(neg, min(len(pos), len(neg)))     # balanced per image
    print(f'windows: train {len(tr)} (pos {sum(i[3] for i in tr)})  '
          f'held-out {len(te)} (pos {sum(i[3] for i in te)})')
    return tr, te


@torch.no_grad()
def score_image(path, m, device):
    """max P(defect) over all sliding windows, and the full map."""
    im = Image.open(path).convert('RGB').resize((S, S))
    x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255)[None].to(device)
    pat = x.unfold(2, WIN, STRIDE).unfold(3, WIN, STRIDE)
    nh, nw = pat.shape[2], pat.shape[3]
    pat = pat.permute(0, 2, 3, 1, 4, 5).reshape(nh * nw, 3, WIN, WIN)
    pat = torch.nn.functional.interpolate(pat, (224, 224), mode='bilinear', align_corners=False)
    pat = (pat - MEAN.to(device)) / STD.to(device)
    sc = []
    for i in range(0, pat.shape[0], 512):
        sc.append(m(pat[i:i + 512]).softmax(1)[:, 1])
    return torch.cat(sc).cpu().numpy().reshape(nh, nw)


@torch.no_grad()
def make_recons(paths, device, out='recon_600'):
    """VQ-VAE reconstructions of real 600px crops: same decoder artefacts, real defect inside."""
    os.makedirs(out, exist_ok=True)
    vq = VQVAE()
    vq.load_state_dict(torch.load('checkpoint/vqvae_560.pt', map_location='cpu'))
    vq = vq.to(device).eval()
    tf = transforms.Compose([transforms.Resize((S, S)), transforms.ToTensor(),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    res = []
    for i in range(0, len(paths), 32):
        chunk = paths[i:i + 32]
        x = torch.stack([tf(Image.open(p).convert('RGB')) for p in chunk]).to(device)
        dec, _ = vq(x)
        dec = ((dec.clamp(-1, 1) + 1) / 2 * 255).round().byte().cpu().permute(0, 2, 3, 1).numpy()
        for p, a in zip(chunk, dec):
            fp = os.path.join(out, os.path.basename(p).replace('.jpg', '.png'))
            Image.fromarray(a).save(fp)
            res.append(fp)
    del vq
    torch.cuda.empty_cache()
    return res


def main():
    device = 'cuda'
    torch.manual_seed(SEED)
    tr, te = build_windows()

    tl = DataLoader(WinDS(tr, True), batch_size=128, shuffle=True, num_workers=8)
    vl = DataLoader(WinDS(te, False), batch_size=128, num_workers=8)

    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m = m.to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    for ep in range(6):
        m.train()
        for x, y in tl:
            opt.zero_grad(); crit(m(x.to(device)), y.to(device)).backward(); opt.step()
        m.eval()
        yt, yp = [], []
        with torch.no_grad():
            for x, y in vl:
                yp += m(x.to(device)).argmax(1).cpu().tolist(); yt += y.tolist()
        yt, yp = np.array(yt), np.array(yp)
        print(f'  ep {ep+1}/6  held-out window acc {(yt==yp).mean():.4f}  '
              f'defect-recall {(yp[yt==1]==1).mean():.4f}  '
              f'clean-recall {(yp[yt==0]==0).mean():.4f}', flush=True)
    torch.save(m.state_dict(), CKPT)

    # ---- calibrate the image-level threshold on HELD-OUT real crops ----
    rng = random.Random(SEED)
    test_boards = set(load_splits()['test_boards'])
    real_te = [p for p in sorted(glob.glob('VOC_PCB/JPEGImages/*.jpg'))
               if (q := parse_filename(os.path.basename(p))) and q['variant'] == 'plain'
               and q['board'] in test_boards]
    real_te = rng.sample(real_te, min(120, len(real_te)))

    null = []
    for p in real_te:
        base = os.path.basename(p)[:-4]
        bs = boxes_256(base)
        sc = score_image(p, m, device)
        cl = []
        for i in range(sc.shape[0]):
            for j in range(sc.shape[1]):
                x0, y0 = j * STRIDE, i * STRIDE
                x1, y1 = x0 + WIN, y0 + WIN
                if not any(not (x1 + 8 <= bx0 or x0 - 8 >= bx1 or
                                y1 + 8 <= by0 or y0 - 8 >= by1) for bx0, by0, bx1, by1 in bs):
                    cl.append(sc[i, j])
        if cl:
            null.append(max(cl))
    null = np.array(null)
    T = float(np.percentile(null, 95))
    print(f'\nimage-level threshold T = {T:.4f}  (95th pct of max-score over provably CLEAN '
          f'windows of held-out real crops => 5% false-positive rate by construction)')

    # ---- the controls and the pools ----
    recon_te = make_recons(real_te, device)
    pools = [
        ('REAL 600px crops (photo, contains a defect)', real_te),
        ('VQ-VAE RECON of those crops (decoder blur + REAL defect)  <- CEILING', recon_te),
        ('GENERATED - demo_pool (ICCE-TW demo)', rng.sample(sorted(glob.glob('demo_pool/*.png')), 120)),
        ('GENERATED - my sample_uncond (same ckpts)', sorted(glob.glob('synth_uncond/u*.png'))),
    ]
    print('\n' + '=' * 86)
    print(f'DEFECT PRESENT?  image-level, calibrated to 5% FPR on real clean board')
    print('=' * 86)
    out = []
    for name, ps in pools:
        mx = np.array([float(score_image(p, m, device).max()) for p in ps])
        frac = float((mx > T).mean())
        out.append({'pool': name, 'n': len(ps), 'frac': frac,
                    'median_max': float(np.median(mx))})
        print(f'  {name:<58} n={len(ps):3d}  {100*frac:5.1f}%   '
              f'(median max {np.median(mx):.3f})')
    json.dump({'threshold': T, 'pools': out},
              open('results/defect_presence_600.json', 'w'), indent=2)
    print('\nwrote results/defect_presence_600.json')


if __name__ == '__main__':
    main()
