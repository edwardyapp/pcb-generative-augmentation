"""Sample from the ORIGINAL UNCONDITIONAL priors + ORIGINAL vqvae_560 — the ICCE-TW
pipeline exactly as published, with no class conditioning and no modifications.

An unconditional prior has no class input, so "N per class" is not a thing you can ask
of it: you draw one undifferentiated pool and the class only exists once a classifier
labels it. That is precisely the point — this pipeline emits an UNLABELED pool. We draw
6 * PER_CLASS samples (the same budget as the conditional run), classify them with the
600px ResNet-18 (matching the 600px crops these priors were trained on), and report the
class distribution that falls out.

Strict state_dict load is asserted: if the n_img_class=0 path had perturbed the original
architecture in any way, this would fail. It does not — the original weights load exactly.
"""
import os
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
def sample_model(model, device, batch, size, temperature, condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64, device=device)
    cache = {}
    for i in range(size[0]):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            row[:, i, j] = torch.multinomial(prob, 1).squeeze(-1)
    return row


def load_prior(path, device):
    """Build from the checkpoint's OWN stored args, unconditional (n_img_class=0),
    and load strict=True so any architecture drift is a hard failure."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    a = ckpt['args']
    if a.hier == 'top':
        model = PixelSNAIL([32, 32], 512, a.channel, 5, 4, a.n_res_block, a.n_res_channel,
                           dropout=a.dropout, n_out_res_block=a.n_out_res_block,
                           n_img_class=0)
    else:
        model = PixelSNAIL([64, 64], 512, a.channel, 5, 4, a.n_res_block, a.n_res_channel,
                           attention=False, dropout=a.dropout, n_cond_res_block=a.n_cond_res_block,
                           cond_res_channel=a.n_res_channel, n_img_class=0)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    assert not missing and not unexpected, \
        f'ARCHITECTURE DRIFT in {path}: missing={missing} unexpected={unexpected}'
    print(f'{path}: loaded strict (0 missing, 0 unexpected)  epoch={a.epoch} data={a.path}')
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vqvae', default='checkpoint/vqvae_560.pt')
    ap.add_argument('--top', default='checkpoint/pixelsnail_top_357.pt')
    ap.add_argument('--bottom', default='checkpoint/pixelsnail_bottom_best.pt')
    ap.add_argument('--clf', default='checkpoint/classifier_A_b100_s0.pt')  # the 600px ResNet-18
    ap.add_argument('--per_class', type=int, default=30)
    ap.add_argument('--batch', type=int, default=45)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--out', default='synth_uncond')
    args = ap.parse_args()
    device = 'cuda'
    total = 6 * args.per_class

    vq = VQVAE()
    vq.load_state_dict(torch.load(args.vqvae, map_location='cpu'))
    vq = vq.to(device).eval()
    top = load_prior(args.top, device)
    bottom = load_prior(args.bottom, device)

    clf = models.resnet18()
    clf.fc = nn.Linear(clf.fc.in_features, 6)
    clf.load_state_dict(torch.load(args.clf, map_location='cpu'))
    clf = clf.to(device).eval()
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    os.makedirs(args.out, exist_ok=True)
    preds, confs, n = [], [], 0
    t0 = time.time()
    while n < total:
        b = min(args.batch, total - n)
        top_s = sample_model(top, device, b, [32, 32], args.temp)
        bot_s = sample_model(bottom, device, b, [64, 64], args.temp, condition=top_s)
        dec = vq.decode_code(top_s, bot_s).clamp(-1, 1)

        x = (dec + 1) / 2
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = (x - mean) / std
        with torch.no_grad():
            logit = clf(x)
        p = logit.argmax(1).cpu().numpy()
        c = logit.softmax(1).max(1).values.cpu().numpy()

        for k in range(b):
            utils.save_image(dec[k], f'{args.out}/u{n+k:03d}_{CLASSES[p[k]]}.png',
                             normalize=True, value_range=(-1, 1))
        utils.save_image(dec, f'{args.out}/grid_batch{n//args.batch:d}.png', nrow=8,
                         normalize=True, value_range=(-1, 1))
        preds.extend(p.tolist()); confs.extend(c.tolist()); n += b
        print(f'  {n}/{total} sampled  ({time.time()-t0:.0f}s)', flush=True)

    preds = np.array(preds); confs = np.array(confs)
    cnt = np.bincount(preds, minlength=6)
    print('\nUNCONDITIONAL POOL — class distribution (600px ResNet-18):')
    for i, c in enumerate(CLASSES):
        bar = '#' * int(40 * cnt[i] / max(1, cnt.max()))
        print(f'  {c:16s} {cnt[i]:4d}/{total}  {100*cnt[i]/total:5.1f}%  {bar}')
    uni = total / 6
    chi2 = float(((cnt - uni) ** 2 / uni).sum())
    print(f'\n  classes never generated: '
          f'{[CLASSES[i] for i in range(6) if cnt[i] == 0] or "none"}')
    print(f'  chi2 vs uniform = {chi2:.1f} (dof 5; >11.07 => p<0.05, distribution NOT uniform)')
    print(f'  mean classifier confidence = {confs.mean():.3f}')
    np.save(f'{args.out}/preds.npy', preds)
    np.save(f'{args.out}/confs.npy', confs)
    print(f'\ntotal {time.time()-t0:.0f}s -> {args.out}/')


if __name__ == '__main__':
    main()
