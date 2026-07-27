"""Is the class-conditioning WIRED correctly? Three tests, independent of sample quality.

A model can be perfectly wired and still generate badly. It can also be BROKEN and produce
numbers that look like weak-but-real conditioning (we have already been fooled by exactly that
kind of thing four times in this project). These tests separate the two questions.

TEST 1  GRADIENT. Do the three class parameters actually receive gradient, in BOTH priors?
        A zero / near-zero grad norm means a dead parameter -- the class label is decorative.

TEST 2  SAMPLING PATH. The cached autoregressive sampler must apply the class bias at EVERY
        step, exactly as training does. The danger: PixelSNAIL caches the top->bottom
        `condition` tensor. If the class term were folded inside that cache branch, it would be
        applied at step 1 and then frozen -- conditioning would silently decay to nothing over
        the remaining 1023 (top) / 4095 (bottom) steps, while still "working" on paper.
        We drive two identical rows with DIFFERENT labels and measure |logit(c=0) - logit(c=1)|
        at every single step. If it collapses to 0 after step 1, the cache is eating the class.

TEST 3  OVERFIT-ONE-BATCH -- the decisive one. Train a FRESH conditional prior on exactly six
        crops, one per class, to zero loss. A correctly wired conditional model MUST then
        reproduce its one memorised example per class perfectly: given class c it has seen
        exactly one code grid, so P(codes | c) is a delta and sampling must return it exactly.
        If it cannot do that, the wiring is broken and no amount of data or epochs can fix it.
"""
import argparse
import os

import numpy as np
import torch
from torch import nn
from torchvision import utils

from dataset import LMDBDataset
from pixelsnail import PixelSNAIL
from vqvae import VQVAE
from pcb_utils import CLASSES

torch.manual_seed(0)


def build(hier, n_img_class=6, channel=128, n_res_block=2, n_res_channel=128):
    """Smaller than the real prior -- this is a wiring test, not a capacity test."""
    if hier == 'top':
        return PixelSNAIL([32, 32], 512, channel, 5, 2, n_res_block, n_res_channel,
                          dropout=0.0, n_out_res_block=0, n_img_class=n_img_class)
    return PixelSNAIL([64, 64], 512, channel, 5, 2, n_res_block, n_res_channel,
                      attention=False, dropout=0.0, n_cond_res_block=3,
                      cond_res_channel=n_res_channel, n_img_class=n_img_class)


# ----------------------------------------------------------------- TEST 1
def test_grad(top_codes, bot_codes, labels, device):
    print('=' * 78)
    print('TEST 1 — does the class embedding receive gradient? (both priors)')
    print('=' * 78)
    crit = nn.CrossEntropyLoss()
    for hier in ('top', 'bottom'):
        m = build(hier).to(device).train()
        m.zero_grad()
        if hier == 'top':
            out, _ = m(top_codes, class_label=labels)
            loss = crit(out, top_codes)
        else:
            out, _ = m(bot_codes, condition=top_codes, class_label=labels)
            loss = crit(out, bot_codes)
        loss.backward()
        print(f'\n  {hier.upper()} prior (loss {loss.item():.4f})')
        ok = True
        for name in ('class_emb', 'class_cond_emb', 'class_out'):
            p = getattr(m, name, None)
            if p is None:
                print(f'    {name:16s} MISSING'); ok = False; continue
            g = p.weight.grad
            if g is None:
                print(f'    {name:16s} grad is None   *** DEAD PARAMETER ***'); ok = False; continue
            gn = g.norm().item()
            per = g.norm(dim=1)
            alive = int((per > 1e-8).sum())
            flag = 'ok' if gn > 1e-6 else '*** ZERO GRAD — DEAD ***'
            if gn <= 1e-6:
                ok = False
            print(f'    {name:16s} |grad| = {gn:.4e}   rows with grad: {alive}/6   {flag}')
        print(f'    => {hier} conditioning params are {"WIRED" if ok else "BROKEN"}')
        del m
        torch.cuda.empty_cache()


# ----------------------------------------------------------------- TEST 2
@torch.no_grad()
def test_sampling_path(device, hier='top', steps=None):
    print('\n' + '=' * 78)
    print(f'TEST 2 — does the CACHED sampler apply the class bias at EVERY step? ({hier})')
    print('=' * 78)
    m = build(hier).to(device).eval()
    size = [32, 32] if hier == 'top' else [64, 64]
    cond = None
    if hier == 'bottom':
        cond = torch.randint(0, 512, (2, 32, 32), device=device)

    # two rows, identical content, DIFFERENT class labels
    row = torch.zeros(2, *size, dtype=torch.int64, device=device)
    lbl = torch.tensor([0, 1], device=device)
    cache = {}
    deltas = []
    n = steps or (size[0] * size[1])
    k = 0
    for i in range(size[0]):
        for j in range(size[1]):
            out, cache = m(row[:, :i + 1, :], condition=cond, cache=cache, class_label=lbl)
            l0, l1 = out[0, :, i, j], out[1, :, i, j]
            deltas.append(float((l0 - l1).abs().max()))
            # keep BOTH rows identical so the ONLY difference is the class label
            s = torch.multinomial(torch.softmax(l0, 0), 1)
            row[:, i, j] = s
            k += 1
            if k >= n:
                break
        if k >= n:
            break
    d = np.array(deltas)
    print(f'  |logit(class=0) − logit(class=1)| across {len(d)} autoregressive steps:')
    print(f'    step 1        : {d[0]:.4f}')
    print(f'    steps 2..10   : mean {d[1:10].mean():.4f}')
    print(f'    last 10 steps : mean {d[-10:].mean():.4f}')
    print(f'    min over ALL  : {d.min():.4f}   (0.0 would mean the class is ignored)')
    dead = int((d < 1e-6).sum())
    print(f'    steps where the class does NOTHING: {dead}/{len(d)}')
    if dead == 0 and d[-10:].mean() > 1e-3:
        print('    => PASS: the class label changes the logits at every step. Cache is NOT')
        print('             swallowing the class bias.')
    else:
        print('    => *** FAIL: the class stops affecting the logits — the cache is eating it ***')
    del m
    torch.cuda.empty_cache()


@torch.no_grad()
def test_determinism(device):
    print('\n' + '=' * 78)
    print('TEST 2b — same class + same seed twice => identical codes? (sampler determinism)')
    print('=' * 78)
    m = build('top').to(device).eval()

    def draw(seed, cls):
        torch.manual_seed(seed)
        row = torch.zeros(1, 32, 32, dtype=torch.int64, device=device)
        cache = {}
        lbl = torch.full((1,), cls, dtype=torch.long, device=device)
        for i in range(32):
            for j in range(32):
                out, cache = m(row[:, :i + 1, :], cache=cache, class_label=lbl)
                p = torch.softmax(out[:, :, i, j], 1)
                row[:, i, j] = torch.multinomial(p, 1).squeeze(-1)
        return row

    a, b = draw(123, 0), draw(123, 0)
    c = draw(123, 1)
    print(f'  same seed, same class  -> identical codes: {bool((a == b).all())}  '
          f'({100*(a==b).float().mean():.1f}% agree)')
    print(f'  same seed, class 0 vs 1-> identical codes: {bool((a == c).all())}  '
          f'({100*(a==c).float().mean():.1f}% agree)   <- should be well under 100%')
    del m
    torch.cuda.empty_cache()


# ----------------------------------------------------------------- TEST 3
def test_overfit(device, epochs=600):
    print('\n' + '=' * 78)
    print('TEST 3 — OVERFIT ONE BATCH (decisive). Six crops, one per class, to zero loss.')
    print('         A correctly-wired conditional model MUST then reproduce each memorised')
    print('         example exactly when asked for that class.')
    print('=' * 78)
    ds = LMDBDataset('lmdb/train_pool_tight')
    # one example per class
    picked, seen = [], set()
    for i in range(len(ds)):
        t, b, l = ds[i]
        if l not in seen and 0 <= l < 6:
            seen.add(l)
            picked.append((t, b, l))
        if len(seen) == 6:
            break
    picked.sort(key=lambda x: x[2])
    top = torch.stack([p[0] for p in picked]).to(device)
    bot = torch.stack([p[1] for p in picked]).to(device)
    lbl = torch.tensor([p[2] for p in picked], device=device)
    print(f'  6 crops, classes {[CLASSES[int(x)] for x in lbl.cpu()]}')

    crit = nn.CrossEntropyLoss()
    res = {}
    for hier, target, cond in (('top', top, None), ('bottom', bot, top)):
        m = build(hier).to(device).train()
        opt = torch.optim.Adam(m.parameters(), lr=3e-4)
        for ep in range(epochs):
            opt.zero_grad()
            out, _ = m(target, condition=cond, class_label=lbl)
            loss = crit(out, target)
            loss.backward()
            opt.step()
            if (ep + 1) % 150 == 0:
                acc = (out.argmax(1) == target).float().mean().item()
                print(f'    {hier} ep {ep+1:4d}  loss {loss.item():.5f}  train acc {acc:.4f}',
                      flush=True)
        m.eval()

        # now SAMPLE each class and see whether we get the memorised grid back
        size = [32, 32] if hier == 'top' else [64, 64]
        with torch.no_grad():
            row = torch.zeros(6, *size, dtype=torch.int64, device=device)
            cache = {}
            c = top if hier == 'bottom' else None
            order = torch.arange(6, device=device)
            for i in range(size[0]):
                for j in range(size[1]):
                    out, cache = m(row[:, :i + 1, :], condition=c, cache=cache, class_label=order)
                    row[:, i, j] = out[:, :, i, j].argmax(1)          # greedy = the mode
        match = (row == target).float().mean(dim=(1, 2))
        res[hier] = (row, match)
        print(f'\n  {hier.upper()}: greedy-sampled codes vs the memorised example, per class')
        for k in range(6):
            print(f'    class {CLASSES[k]:16s} {100*match[k]:6.2f}% of codes reproduced exactly')
        print(f'    MEAN {100*match.mean():.2f}%')
        del m
        torch.cuda.empty_cache()

    ok = res['top'][1].mean() > 0.95 and res['bottom'][1].mean() > 0.80
    print('\n  ' + '-' * 70)
    if ok:
        print('  => PASS. The model reproduces its one memorised example per class. The class')
        print('     conditioning is CORRECTLY WIRED end to end (train AND cached sampling).')
        print('     Poor sample quality is therefore a MODELLING/DATA problem, not a bug.')
    else:
        print('  => *** FAIL. The model cannot reproduce its own memorised example when asked')
        print('     by class. THE WIRING IS BROKEN — more data or epochs cannot fix this. ***')

    # decode for the eye
    try:
        vq = VQVAE()
        vq.load_state_dict(torch.load('checkpoint/vqvae_tight_560.pt', map_location='cpu'))
        vq = vq.to(device).eval()
        with torch.no_grad():
            real = vq.decode_code(top, bot).clamp(-1, 1)
            fake = vq.decode_code(res['top'][0], res['bottom'][0]).clamp(-1, 1)
        os.makedirs('figures', exist_ok=True)
        utils.save_image(torch.cat([real, fake], 0), 'figures/fig13_overfit_one_batch.png',
                         nrow=6, normalize=True, value_range=(-1, 1))
        print('\n  wrote figures/fig13_overfit_one_batch.png  (top row = the 6 memorised crops,')
        print('  bottom row = what the model generates when asked for each class)')
    except Exception as e:
        print(f'  (decode skipped: {e})')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=600)
    ap.add_argument('--skip', default='')
    args = ap.parse_args()
    device = 'cuda'

    # one example PER CLASS -- otherwise the gradient test only exercises one embedding row
    ds = LMDBDataset('lmdb/train_pool_tight')
    picked, seen = [], set()
    for i in range(len(ds)):
        t_, b_, l_ = ds[i]
        if 0 <= l_ < 6 and l_ not in seen:
            seen.add(l_)
            picked.append((t_, b_, l_))
        if len(seen) == 6:
            break
    picked.sort(key=lambda x: x[2])
    top = torch.stack([p[0] for p in picked]).to(device)
    bot = torch.stack([p[1] for p in picked]).to(device)
    lbl = torch.tensor([p[2] for p in picked], device=device)

    if '1' not in args.skip:
        test_grad(top, bot, lbl, device)
    if '2' not in args.skip:
        test_sampling_path(device, 'top')
        test_sampling_path(device, 'bottom', steps=200)
        test_determinism(device)
    if '3' not in args.skip:
        test_overfit(device, args.epochs)


if __name__ == '__main__':
    main()
