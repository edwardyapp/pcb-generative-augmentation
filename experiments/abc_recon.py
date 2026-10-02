"""A/B/C downstream experiment with VQ-VAE RECONSTRUCTIONS standing in for generated samples.

This is the honest UPPER BOUND on the whole generative arm. Reconstructions of real tight
crops are (a) labelled by construction and (b) classify at 0.918 macro-F1 — i.e. they are
about as good as synthetic data can possibly get, because they ARE real data round-tripped
through the same decoder any generator must use. No prior can beat them.

If reconstructions do not lift the classifier at the 10% and 25% budgets, then no generator
will, and the entire generative thesis is dead independently of whether our prior converges.

PRE-REGISTERED, before seeing any result:
  * Filter (Condition C) = the Condition-A model AT THE SAME BUDGET AND SEED, trained inline
    here, so the no-leakage rule ("the filter classifier at budget b must be exactly the
    baseline classifier trained at budget b") holds by construction. Keep a synthetic sample
    iff that model predicts its true label. No confidence threshold.
  * Two synthetic pools, because the distinction is the whole ballgame:
      - SELF: reconstructions of ONLY the budget-b crops you actually hold. No label you do
        not own. This is the honest augmentation value.
      - POOL: reconstructions of the FULL train pool (all 2149). This is what our prior
        actually saw — it was trained on the full train pool regardless of budget — so it is
        the true ceiling for OUR pipeline, but it LEAKS labels you would not have at a 10%
        budget. Reported separately and never conflated.
  * Success = B or C beats A by more than the seed-to-seed std at the 10% or 25% budget.
  * Everything else identical to Condition A: same recipe, same board split, same 3 seeds,
    same 20 epochs, final-epoch metric (no test-set model selection).
"""
import sys, pathlib  # repo layout (README.md): src/, experiments/, analysis/ importable from anywhere
sys.path[1:1] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('src', 'experiments', 'analysis')]
import os
import json
import copy
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image

from vqvae import VQVAE
from pcb_utils import CLASSES, load_splits
from train_classifier import (set_seed, read_manifest, CropDS, build_train_items,
                              build_test_items, macro_f1, evaluate)

RECON_DIR = 'recon_pool'
VQ_CKPT = 'checkpoint/vqvae_tight_560.pt'
MANIFEST = 'results/manifest_tight.csv'
BUDGETS = [0.10, 0.25, 0.50, 1.00]
SEEDS = [0, 1, 2]

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), transforms.RandomRotation(15),
    transforms.ToTensor(), NORM])
TEST_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])
VQ_TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])


@torch.no_grad()
def build_recon_pool(rows, device):
    """Round-trip every TRAIN tight crop through the VQ-VAE. Test boards never touched."""
    os.makedirs(RECON_DIR, exist_ok=True)
    train = [r for r in rows if r['split'] == 'train']
    test_boards = set(load_splits()['test_boards'])
    for r in train:
        assert r['board_id'] not in test_boards, f"LEAK: board {r['board_id']} in recon pool"

    out = [(os.path.join(RECON_DIR, os.path.basename(r['crop_path'])), int(r['class_idx']))
           for r in train]
    if all(os.path.exists(p) for p, _ in out):
        print(f'recon pool already built: {len(out)} crops')
        return out, {r['crop_path']: o[0] for r, o in zip(train, out)}

    vq = VQVAE()
    vq.load_state_dict(torch.load(VQ_CKPT, map_location='cpu'))
    vq = vq.to(device).eval()
    B = 64
    for i in range(0, len(train), B):
        chunk = train[i:i + B]
        x = torch.stack([VQ_TF(Image.open(r['crop_path']).convert('RGB')) for r in chunk]).to(device)
        dec, _ = vq(x)
        dec = ((dec.clamp(-1, 1) + 1) / 2 * 255).round().byte().cpu().permute(0, 2, 3, 1).numpy()
        for r, a in zip(chunk, dec):
            Image.fromarray(a).save(os.path.join(RECON_DIR, os.path.basename(r['crop_path'])))
    print(f'built recon pool: {len(out)} crops -> {RECON_DIR}/')
    return out, {r['crop_path']: o[0] for r, o in zip(train, out)}


def train_clf(items, test_loader, device, seed, epochs=20, batch=64, lr=1e-4):
    set_seed(seed)
    tl = DataLoader(CropDS(items, TRAIN_TF), batch_size=batch, shuffle=True,
                    num_workers=8, drop_last=False)
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 6)
    m = m.to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        m.train()
        for x, y in tl:
            opt.zero_grad()
            crit(m(x.to(device)), y.to(device)).backward()
            opt.step()
    f1, acc, per = evaluate(m, test_loader, device)
    return m, f1, acc, per


@torch.no_grad()
def filter_pool(model, pool, device, batch=128):
    """Condition C: keep a synthetic crop iff the budget-matched baseline predicts its label."""
    model.eval()
    kept = []
    for i in range(0, len(pool), batch):
        chunk = pool[i:i + batch]
        x = torch.stack([TEST_TF(Image.open(p).convert('RGB')) for p, _ in chunk]).to(device)
        pred = model(x).argmax(1).cpu().numpy()
        kept += [it for it, p in zip(chunk, pred) if p == it[1]]
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--out', default='results/abc_recon.json')
    args = ap.parse_args()
    device = 'cuda'

    rows = read_manifest(MANIFEST)
    recon_pool, recon_of = build_recon_pool(rows, device)
    test_items = build_test_items(rows)
    vl = DataLoader(CropDS(test_items, TEST_TF), batch_size=64, shuffle=False, num_workers=8)
    print(f'test crops: {len(test_items)}  (boards {load_splits()["test_boards"]})\n')

    results = []
    for b in BUDGETS:
        for s in SEEDS:
            real = build_train_items(rows, b, s)
            self_pool = [(recon_of[p], y) for p, y in real]          # recon of ONLY what you hold
            full_pool = recon_pool                                    # recon of the whole train pool

            # --- A: real only. This model IS the filter (no-leakage rule). ---
            fmodel, f1a, acca, pera = train_clf(real, vl, device, s, args.epochs)

            kept_self = filter_pool(fmodel, self_pool, device)
            kept_full = filter_pool(fmodel, full_pool, device)

            runs = {
                'B_self': real + self_pool,
                'C_self': real + kept_self,
                'B_pool': real + full_pool,
                'C_pool': real + kept_full,
            }
            rec = {'budget': b, 'seed': s, 'n_real': len(real),
                   'A': {'macro_f1': f1a, 'accuracy': acca, 'per_class_f1': pera},
                   'keep_rate_self': len(kept_self) / max(1, len(self_pool)),
                   'keep_rate_pool': len(kept_full) / max(1, len(full_pool))}
            line = (f'budget {int(b*100):3d}% seed {s} | n_real {len(real):4d} | '
                    f'A {f1a:.3f}')
            for name, items in runs.items():
                _, f1, acc, per = train_clf(items, vl, device, s, args.epochs)
                rec[name] = {'macro_f1': f1, 'accuracy': acc, 'per_class_f1': per,
                             'n_train': len(items)}
                line += f' | {name} {f1:.3f}'
            line += (f' | keep self {rec["keep_rate_self"]*100:.0f}% '
                     f'pool {rec["keep_rate_pool"]*100:.0f}%')
            print(line, flush=True)
            results.append(rec)

    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2)

    # ---------------- summary ----------------
    print('\n' + '=' * 92)
    print('A/B/C with VQ-VAE RECONSTRUCTIONS as synthetic  (macro-F1, mean +/- std over 3 seeds)')
    print('=' * 92)
    conds = ['A', 'B_self', 'C_self', 'B_pool', 'C_pool']
    print(f'{"budget":>7} ' + ' '.join(f'{c:>14}' for c in conds) + '   verdict @ that budget')
    for b in BUDGETS:
        cell = {c: np.array([r[c]['macro_f1'] for r in results if r['budget'] == b]) for c in conds}
        a_m, a_s = cell['A'].mean(), cell['A'].std()
        row = f'{int(b*100):6d}% '
        for c in conds:
            row += f'{cell[c].mean():>8.3f}±{cell[c].std():.3f} '
        best = max(conds[1:], key=lambda c: cell[c].mean())
        lift = cell[best].mean() - a_m
        row += f'  best={best} {lift:+.3f} ' + ('LIFT' if lift > a_s else 'no lift (< A seed-std)')
        print(row)
    print('\nInterpretation: B_pool/C_pool are the RECONSTRUCTION ceiling (recon of the full train pool —')
    print('labels you would not own at a 10% budget). No generator is involved; a generator trained on')
    print('the full pool delivers +0.002 at 10% (results/abc_leaky_b*.json, AUDIT.md section 7).')
    print('B_self/C_self are the honest no-leak augmentation value. If even B_pool fails to lift A')
    print('at 10%/25%, no generator built on this pipeline can help.')
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
