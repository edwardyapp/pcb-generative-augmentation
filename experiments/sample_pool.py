"""Sample a LABELLED synthetic pool from a budget-restricted generator.

The label is the class the prior was conditioned on -- that is the whole premise of the
conditional pipeline, and the gate has already established (or not) that it is meaningful.
Condition C will later re-check each sample with the budget-b classifier; that filter is a
separate step and is NOT applied here.

All six classes are sampled in ONE autoregressive pass (chunked to fit VRAM). Wall-clock is
dominated by the 1024+4096 sequential steps, not by batch size, so batching the classes
together is ~6x cheaper than looping over them.
"""
import sys, pathlib  # repo layout (README.md): src/, experiments/, analysis/ importable from anywhere
sys.path[1:1] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('src', 'experiments', 'analysis')]
import os
import time
import argparse

import torch
from torchvision import utils

from vqvae import VQVAE
from pcb_utils import CLASSES
from cond_check import sample_model, load_prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vqvae', required=True)
    ap.add_argument('--top', required=True)
    ap.add_argument('--bottom', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--per_class', type=int, default=60)
    ap.add_argument('--chunk', type=int, default=36)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--uncond', action='store_true',
                    help='priors have n_img_class=0: sample with NO class label at all. Every '
                         'sample is still a defect crop by construction (the prior was trained '
                         'only on defect crops), which is exactly what the binary track needs.')
    args = ap.parse_args()
    device = 'cuda'

    vq = VQVAE()
    vq.load_state_dict(torch.load(args.vqvae, map_location='cpu'))
    vq = vq.to(device).eval()
    top, bottom = load_prior(args.top, device), load_prior(args.bottom, device)

    os.makedirs(args.out, exist_ok=True)
    labels = torch.arange(6, device=device).repeat_interleave(args.per_class)
    t0, n = time.time(), 0
    for i in range(0, labels.shape[0], args.chunk):
        lbl = labels[i:i + args.chunk]
        cl = None if args.uncond else lbl
        top_s = sample_model(top, device, [32, 32], args.temp, cl, batch=lbl.shape[0])
        bot_s = sample_model(bottom, device, [64, 64], args.temp, cl, condition=top_s)
        dec = vq.decode_code(top_s, bot_s).clamp(-1, 1)
        for k in range(dec.shape[0]):
            c = 'defect' if args.uncond else CLASSES[int(lbl[k])]
            utils.save_image(dec[k], f'{args.out}/{c}_{n:04d}.png',
                             normalize=True, value_range=(-1, 1))
            n += 1
        print(f'  {n}/{labels.shape[0]} ({time.time()-t0:.0f}s)', flush=True)
    print(f'wrote {n} labelled synthetic crops -> {args.out}/  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
