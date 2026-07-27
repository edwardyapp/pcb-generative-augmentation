"""Conditioning check: sample per class from the tight conditional priors, decode
through the tight VQ-VAE, classify with the real-crop tight ResNet-18, and measure
whether predicted class == conditioned class.

The class bias is applied inside forward() to `out = horizontal + vertical`, which is
recomputed every autoregressive step (the cache only short-circuits the top->bottom
`condition` path), so class conditioning is applied identically to training here.
"""
import os
import time
import argparse

import numpy as np
import torch
from torch import nn
from torchvision import utils, models
from tqdm import tqdm

from vqvae import VQVAE
from pixelsnail import PixelSNAIL
from pcb_utils import CLASSES

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def sample_model(model, device, batch, size, temperature, class_label, condition=None):
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
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    a = ckpt['args']
    if a.hier == 'top':
        model = PixelSNAIL([32, 32], 512, a.channel, 5, 4, a.n_res_block, a.n_res_channel,
                           dropout=a.dropout, n_out_res_block=a.n_out_res_block,
                           n_img_class=a.n_img_class)
    else:
        model = PixelSNAIL([64, 64], 512, a.channel, 5, 4, a.n_res_block, a.n_res_channel,
                           attention=False, dropout=a.dropout, n_cond_res_block=a.n_cond_res_block,
                           cond_res_channel=a.n_res_channel, n_img_class=a.n_img_class)
    model.load_state_dict(ckpt['model'])
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vqvae', default='checkpoint/vqvae_tight_560.pt')
    ap.add_argument('--top', default='checkpoint/pixelsnail_condtight_top_060.pt')
    ap.add_argument('--bottom', default='checkpoint/pixelsnail_condtight_bottom_060.pt')
    ap.add_argument('--clf', default='checkpoint/classifier_Atight_b100_s0.pt')
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--max_classes', type=int, default=6)
    ap.add_argument('--out', default='synth_check')
    args = ap.parse_args()
    device = 'cuda'

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
    conf = np.zeros((6, 6), int)  # [conditioned, predicted]
    for c in range(min(args.max_classes, 6)):
        t0 = time.time()
        lbl = torch.full((args.n,), c, dtype=torch.long, device=device)
        top_s = sample_model(top, device, args.n, [32, 32], args.temp, lbl)
        bot_s = sample_model(bottom, device, args.n, [64, 64], args.temp, lbl, condition=top_s)
        dec = vq.decode_code(top_s, bot_s).clamp(-1, 1)
        utils.save_image(dec, f'{args.out}/grid_{CLASSES[c]}.png', nrow=8,
                         normalize=True, value_range=(-1, 1))
        x = (dec + 1) / 2
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = (x - mean) / std
        with torch.no_grad():
            pred = clf(x).argmax(1).cpu().numpy()
        for p in pred:
            conf[c, p] += 1
        print(f'cond={CLASSES[c]:<16} match={100*(pred==c).mean():5.1f}%  '
              + 'pred: ' + ' '.join(f'{CLASSES[k][:4]}={int((pred==k).sum())}' for k in range(6))
              + f'  ({time.time()-t0:.0f}s)', flush=True)

    tot = conf.sum()
    if tot:
        print(f'\nCONDITIONING CONSISTENCY (pred==conditioned): '
              f'{np.trace(conf)}/{tot} = {100*np.trace(conf)/tot:.1f}%  (chance {100/6:.1f}%)')
        print('confusion [row=conditioned, col=predicted]:')
        print('            ' + ' '.join(f'{c[:6]:>6}' for c in CLASSES))
        for i, c in enumerate(CLASSES):
            print(f'{c:<12}' + ' '.join(f'{conf[i,j]:>6}' for j in range(6)))


if __name__ == '__main__':
    main()
