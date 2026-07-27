"""Conditioning check at a given epoch — batched, so it is cheap enough to run at six
epochs instead of once at the end.

The old check sampled each class in its OWN autoregressive pass (6 passes). Wall-clock is
dominated by the 1024+4096 sequential steps, NOT by batch size, so batching all six classes
into one pass with mixed class labels costs ~1 pass instead of 6.

PRE-REGISTERED metrics (fixed before any result was seen), because a pooled consistency
number is actively misleading when the model collapses onto one class:
  * consistency        - diag/total. The naive number.
  * per-class          - diag_i / n_i, reported for every class, always.
  * collapse_index     - the largest share of ALL predictions taken by any single class.
                         1/6 = 0.167 means perfectly spread. We flag COLLAPSE at > 0.40.
  * consistency_ex     - consistency recomputed with the collapse class's ROW removed.
                         At epoch 80 this was the tell: 22.4% pooled, but 13.1% once the
                         'spur' row came out - i.e. BELOW the 16.7% chance line.
  * cramers_v          - association between conditioned and predicted class. 0 = the label
                         does nothing. This is the collapse-proof summary: unlike consistency
                         it cannot be inflated by the model dumping everything in one class.
"""
import os
import json
import time
import argparse

import numpy as np
import torch
from torch import nn
from torchvision import utils, models

from vqvae import VQVAE
from pixelsnail import PixelSNAIL
from pcb_utils import CLASSES

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def sample_model(model, device, size, temperature, class_label, condition=None, batch=None):
    # class_label=None is the unconditional path (n_img_class=0 priors): the batch size
    # must then come from `condition` or be given explicitly.
    if class_label is not None:
        batch = class_label.shape[0]
    elif condition is not None:
        batch = condition.shape[0]
    if batch is None:
        raise ValueError('unconditional top-level sampling needs an explicit batch=')
    row = torch.zeros(batch, *size, dtype=torch.int64, device=device)
    cache = {}
    for i in range(size[0]):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache,
                               class_label=class_label)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            row[:, i, j] = torch.multinomial(prob, 1).squeeze(-1)
    return row


def load_prior(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    a = ck['args']
    if a.hier == 'top':
        m = PixelSNAIL([32, 32], 512, a.channel, 5, 4, a.n_res_block, a.n_res_channel,
                       dropout=a.dropout, n_out_res_block=a.n_out_res_block,
                       n_img_class=a.n_img_class)
    else:
        m = PixelSNAIL([64, 64], 512, a.channel, 5, 4, a.n_res_block, a.n_res_channel,
                       attention=False, dropout=a.dropout, n_cond_res_block=a.n_cond_res_block,
                       cond_res_channel=a.n_res_channel, n_img_class=a.n_img_class)
    m.load_state_dict(ck['model'])
    return m.to(device).eval()


def cramers_v(conf):
    """Association between conditioned (rows) and predicted (cols). 0 => label does nothing."""
    n = conf.sum()
    if n == 0:
        return 0.0
    row, col = conf.sum(1, keepdims=True), conf.sum(0, keepdims=True)
    exp = row @ col / n
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum(np.where(exp > 0, (conf - exp) ** 2 / exp, 0.0))
    k = min(conf.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


def analyse(conf):
    n = int(conf.sum())
    diag = int(np.trace(conf))
    per = {CLASSES[i]: float(conf[i, i] / max(1, conf[i].sum())) for i in range(6)}
    colshare = conf.sum(0) / max(1, n)
    ci = int(colshare.argmax())
    collapse = float(colshare[ci])
    keep = [i for i in range(6) if i != ci]
    sub = conf[keep]
    diag_ex = sum(int(conf[i, i]) for i in keep)
    return {
        'n': n,
        'consistency': diag / max(1, n),
        'per_class': per,
        'collapse_class': CLASSES[ci],
        'collapse_index': collapse,
        'collapsed': bool(collapse > 0.40),
        'consistency_ex': diag_ex / max(1, int(sub.sum())) if sub.sum() else 0.0,
        'cramers_v': cramers_v(conf.astype(float)),
        'confusion': conf.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', required=True)
    ap.add_argument('--bottom', required=True)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--tag', default='condtight3')
    ap.add_argument('--vqvae', default='checkpoint/vqvae_tight_560.pt')
    ap.add_argument('--clf', default='checkpoint/classifier_Atight_b100_s0.pt')
    ap.add_argument('--per_class', type=int, default=12)
    ap.add_argument('--chunk', type=int, default=36)   # cap so it fits in 24GB (4090)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--out', default='results/cond_checks.jsonl')
    ap.add_argument('--grid_dir', default=None)
    args = ap.parse_args()
    device = 'cuda'

    vq = VQVAE()
    vq.load_state_dict(torch.load(args.vqvae, map_location='cpu'))
    vq = vq.to(device).eval()
    top, bottom = load_prior(args.top, device), load_prior(args.bottom, device)

    clf = models.resnet18()
    clf.fc = nn.Linear(clf.fc.in_features, 6)
    clf.load_state_dict(torch.load(args.clf, map_location='cpu'))
    clf = clf.to(device).eval()
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    labels = torch.arange(6, device=device).repeat_interleave(args.per_class)
    conf = np.zeros((6, 6), int)
    t0 = time.time()
    for i in range(0, labels.shape[0], args.chunk):
        lbl = labels[i:i + args.chunk]
        top_s = sample_model(top, device, [32, 32], args.temp, lbl)
        bot_s = sample_model(bottom, device, [64, 64], args.temp, lbl, condition=top_s)
        dec = vq.decode_code(top_s, bot_s).clamp(-1, 1)
        if args.grid_dir:
            os.makedirs(args.grid_dir, exist_ok=True)
            utils.save_image(dec, f'{args.grid_dir}/ep{args.epoch:03d}_chunk{i//args.chunk}.png',
                             nrow=6, normalize=True, value_range=(-1, 1))
        x = (dec + 1) / 2
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        with torch.no_grad():
            pred = clf((x - mean) / std).argmax(1).cpu().numpy()
        for c, p in zip(lbl.cpu().numpy(), pred):
            conf[c, p] += 1
        print(f'  chunk {i//args.chunk}: {i+lbl.shape[0]}/{labels.shape[0]} '
              f'({time.time()-t0:.0f}s)', flush=True)

    r = analyse(conf)
    r.update({'epoch': args.epoch, 'tag': args.tag, 'temp': args.temp,
              'top': args.top, 'bottom': args.bottom, 'secs': time.time() - t0})

    print(f'\n=== {args.tag} epoch {args.epoch} ===')
    print(f'consistency      {100*r["consistency"]:5.1f}%   (chance 16.7%)')
    print(f'collapse         {r["collapse_class"]} takes {100*r["collapse_index"]:.1f}% of ALL '
          f'predictions  ' + ('*** COLLAPSED ***' if r['collapsed'] else '(ok)'))
    print(f'consistency_ex   {100*r["consistency_ex"]:5.1f}%   (collapse row removed; '
          f'{"BELOW" if r["consistency_ex"] < 1/6 else "above"} chance)')
    print(f"cramer's V       {r['cramers_v']:.3f}   (0 = the class label does nothing)")
    print('per-class consistency:')
    for c in CLASSES:
        v = r['per_class'][c]
        print(f'   {c:16s} {100*v:5.1f}%  ' + ('' if v > 1/6 else '<- at/below chance'))
    print('confusion [row=conditioned, col=predicted]:')
    for i, c in enumerate(CLASSES):
        print(f'   {c:16s}' + ' '.join(f'{conf[i,j]:4d}' for j in range(6)))

    os.makedirs('results', exist_ok=True)
    with open(args.out, 'a') as fh:
        fh.write(json.dumps(r) + '\n')
    print(f'\nappended -> {args.out}')


if __name__ == '__main__':
    main()
