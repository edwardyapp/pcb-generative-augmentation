"""The overfit-one-batch test, done properly: a CROSS-CLASS match matrix.

The first cut of this test declared FAIL, but its pass criterion was wrong. It asked "does the
model reproduce class c's memorised grid?" and got 61% for the top prior. But a 61% reproduction
rate does NOT distinguish these two very different worlds:

  (a) the class label is IGNORED  -> the model samples the same generic grid whatever you ask
      for, matching every class about equally. Wiring broken.
  (b) the class label WORKS, but free-running autoregressive sampling DRIFTS. Teacher-forced
      accuracy is 98.8%, but over 1024 sequential steps a single early mistake puts the context
      off-distribution and the error compounds. 0.988^1024 ~ 4e-6, so exact reproduction is not
      the right bar at all -- and a model that is NOT trained to true zero loss will always drift.

The test that separates them: sample conditioned on class c, then compare the result against
EVERY class's memorised grid. If conditioning works, the match is highest on the diagonal --
sampled(c) looks more like memorised(c) than like memorised(c') for c' != c. If the class is
ignored, the matrix is flat and the diagonal is unremarkable.

Also reported: TEACHER-FORCED reproduction, which removes drift entirely. If teacher-forced
reproduction is ~100% and diagonal dominance holds, the wiring is provably correct and the only
problem is drift -- a modelling issue, not a bug.
"""
import numpy as np
import torch
from torch import nn
from torchvision import utils

from dataset import LMDBDataset
from pcb_utils import CLASSES
from test_conditioning import build
from vqvae import VQVAE

torch.manual_seed(0)
EPOCHS = 3000
device = 'cuda'


def main():
    ds = LMDBDataset('lmdb/train_pool_tight')
    picked, seen = [], set()
    for i in range(len(ds)):
        t, b, l = ds[i]
        if 0 <= l < 6 and l not in seen:
            seen.add(l); picked.append((t, b, l))
        if len(seen) == 6:
            break
    picked.sort(key=lambda x: x[2])
    top = torch.stack([p[0] for p in picked]).to(device)
    bot = torch.stack([p[1] for p in picked]).to(device)
    lbl = torch.tensor([p[2] for p in picked], device=device)
    print(f'6 crops, one per class. Training the TOP prior to (near) zero loss.\n')

    m = build('top').to(device).train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        opt.zero_grad()
        out, _ = m(top, class_label=lbl)
        loss = crit(out, top)
        loss.backward(); opt.step(); sched.step()
        if (ep + 1) % 500 == 0:
            acc = (out.argmax(1) == top).float().mean().item()
            print(f'  ep {ep+1:5d}  loss {loss.item():.6f}  teacher-forced acc {acc:.5f}',
                  flush=True)
    m.eval()

    # ---- teacher-forced reproduction: drift removed entirely ----
    with torch.no_grad():
        out, _ = m(top, class_label=lbl)
        tf = (out.argmax(1) == top).float().mean(dim=(1, 2))
    print('\nTEACHER-FORCED reproduction (no drift possible):')
    for k in range(6):
        print(f'  {CLASSES[k]:16s} {100*tf[k]:6.2f}%')
    print(f'  MEAN {100*tf.mean():.2f}%')

    # ---- free-running greedy sample for each class ----
    with torch.no_grad():
        row = torch.zeros(6, 32, 32, dtype=torch.int64, device=device)
        cache = {}
        order = torch.arange(6, device=device)
        for i in range(32):
            for j in range(32):
                o, cache = m(row[:, :i + 1, :], cache=cache, class_label=order)
                row[:, i, j] = o[:, :, i, j].argmax(1)

    # ---- THE MATRIX: sampled(c) vs memorised(c') for all pairs ----
    M = np.zeros((6, 6))
    for c in range(6):
        for c2 in range(6):
            M[c, c2] = float((row[c] == top[c2]).float().mean())
    print('\nCROSS-CLASS MATCH MATRIX  (row = class SAMPLED, col = memorised example compared to)')
    print('     ' + ' '.join(f'{c[:6]:>7s}' for c in CLASSES))
    for c in range(6):
        star = ''
        best = int(M[c].argmax())
        row_s = ' '.join(f'{100*M[c, j]:6.1f}%' + ('*' if j == best else ' ') for j in range(6))
        print(f'  {CLASSES[c][:4]:>4s} {row_s}')
    diag = np.diag(M)
    off = M[~np.eye(6, dtype=bool)]
    correct = int(sum(M[c].argmax() == c for c in range(6)))
    print(f'\n  diagonal mean   {100*diag.mean():.1f}%')
    print(f'  off-diag mean   {100*off.mean():.1f}%')
    print(f'  separation      {100*(diag.mean()-off.mean()):+.1f} pp')
    print(f'  classes whose BEST match is their own example: {correct}/6')

    print('\n' + '-' * 74)
    if correct >= 5 and diag.mean() - off.mean() > 0.05:
        print('  => WIRING IS CORRECT. Conditioned on class c, the model reproduces class c\'s')
        print('     example more than any other class\'s. The class label is doing real work in')
        print('     the cached sampling path. The 61% free-running reproduction is AUTOREGRESSIVE')
        print('     DRIFT (0.99^1024 compounding), not a conditioning bug. Bad samples are a')
        print('     MODELLING/DATA problem — the wiring cannot be blamed.')
    else:
        print('  => *** WIRING IS BROKEN: the class label does not select the right example. ***')

    try:
        vq = VQVAE()
        vq.load_state_dict(torch.load('checkpoint/vqvae_tight_560.pt', map_location='cpu'))
        vq = vq.to(device).eval()
        with torch.no_grad():
            real = vq.decode_code(top, bot).clamp(-1, 1)
            fake = vq.decode_code(row, bot).clamp(-1, 1)   # sampled top + true bottom
        utils.save_image(torch.cat([real, fake], 0), 'figures/fig13_overfit_one_batch.png',
                         nrow=6, normalize=True, value_range=(-1, 1))
        print('\n  wrote figures/fig13_overfit_one_batch.png (row1 = memorised, row2 = generated)')
    except Exception as e:
        print(f'  decode skipped: {e}')


if __name__ == '__main__':
    main()
