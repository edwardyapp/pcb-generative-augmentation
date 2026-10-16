"""BINARY track, Condition A: defect vs no-defect, across budgets. The gate for the whole track.

The 6-way task may simply be the wrong task. A binary detector is what an AOI line actually
needs, and it is a far easier target -- which is exactly the risk: if the binary baseline is
already near-perfect with 10% of the data, there is NO HEADROOM for synthetic data to fill, and
the entire generative track is dead on arrival. Better to learn that in ten minutes than after a
week of sampling.

POSITIVES  the tight crops (a defect fills ~22% of the frame).
NEGATIVES  windows from THE SAME 600px crops, same boards, same 128px-native -> 256 recipe,
           rejection-sampled to miss every annotated bbox by a margin. Real substrate, real
           traces, real pads -- the ONLY difference is the presence of a defect, so the model
           cannot cheat on colour, texture or board identity.
Same board split (06/09 held out), same ResNet-18 recipe as everywhere else, 3 seeds.
Budgets subsample positives AND negatives at the same rate.

GATE: if macro-F1 at the 10% budget is already >= 0.95, there is no room to improve and we stop.
"""
import sys, pathlib  # repo layout (README.md): src/, experiments/, analysis/ importable from anywhere
sys.path[1:1] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('src', 'experiments', 'analysis')]
import csv
import json
import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from pcb_utils import load_splits
from train_classifier import CropDS, set_seed
from abc_recon import TRAIN_TF, TEST_TF

BUDGETS = [0.10, 0.25, 0.50, 1.00]
SEEDS = [0, 1, 2]
GATE = 0.95


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def pools():
    pos = load('results/manifest_tight_perbbox.csv')
    neg = load('results/manifest_nodefect.csv')
    tb = set(load_splits()['test_boards'])
    for r in pos + neg:
        if r['split'] == 'train':
            assert r['board_id'] not in tb, f'LEAK: test board {r["board_id"]} in train split'
    tr = [(r['crop_path'], 1) for r in pos if r['split'] == 'train'] + \
         [(r['crop_path'], 0) for r in neg if r['split'] == 'train']
    # balanced held-out set: equal positives and negatives
    te_p = [(r['crop_path'], 1) for r in pos if r['split'] == 'test']
    te_n = [(r['crop_path'], 0) for r in neg if r['split'] == 'test']
    te_n = random.Random(0).sample(te_n, min(len(te_p), len(te_n)))
    overlap = {p for p, _ in tr} & {p for p, _ in te_p + te_n}
    assert not overlap, f'LEAK: {len(overlap)} crops in both train and test'
    return tr, te_p + te_n


def subsample(items, budget, seed):
    """Stratified by label, so the positive:negative ratio is held fixed across budgets."""
    by = defaultdict(list)
    for p, y in items:
        by[y].append((p, y))
    rng = random.Random(seed)
    out = []
    for y in sorted(by):
        lst = sorted(by[y])
        rng.shuffle(lst)
        k = len(lst) if budget >= 1.0 else max(1, int(round(len(lst) * budget)))
        out += lst[:k]
    return out


def metrics(yt, yp, ps):
    yt, yp, ps = np.array(yt), np.array(yp), np.array(ps)
    f1s = []
    for c in (0, 1):
        tp = int(((yp == c) & (yt == c)).sum())
        fp = int(((yp == c) & (yt != c)).sum())
        fn = int(((yp != c) & (yt == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.
        rc = tp / (tp + fn) if tp + fn else 0.
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.)
    # AUC by rank
    order = np.argsort(ps)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(ps) + 1)
    npos, nneg = (yt == 1).sum(), (yt == 0).sum()
    auc = ((ranks[yt == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else 0.5
    return float(np.mean(f1s)), float((yt == yp).mean()), float(auc), f1s


def run(items, test_loader, device, seed, epochs=20):
    set_seed(seed)
    tl = DataLoader(CropDS(items, TRAIN_TF), batch_size=64, shuffle=True, num_workers=8)
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m = m.to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        m.train()
        for x, y in tl:
            opt.zero_grad(); crit(m(x.to(device)), y.to(device)).backward(); opt.step()
    m.eval()
    yt, yp, ps = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            o = m(x.to(device))
            ps += o.softmax(1)[:, 1].cpu().tolist()
            yp += o.argmax(1).cpu().tolist()
            yt += y.tolist()
    return m, metrics(yt, yp, ps)


def main():
    device = 'cuda'
    tr, te = pools()
    print(f'positives+negatives: train {len(tr)} (pos {sum(y for _, y in tr)}), '
          f'held-out {len(te)} (pos {sum(y for _, y in te)}) on boards '
          f'{load_splits()["test_boards"]}\n')
    vl = DataLoader(CropDS(te, TEST_TF), batch_size=64, num_workers=8)

    rows = []
    print(f'{"budget":>7} {"n_train":>8}  {"macro-F1":>16} {"accuracy":>16} {"AUC":>16}')
    for b in BUDGETS:
        f1s, accs, aucs = [], [], []
        for s in SEEDS:
            items = subsample(tr, b, s)
            _, (f1, acc, auc, per) = run(items, vl, device, s)
            f1s.append(f1); accs.append(acc); aucs.append(auc)
            rows.append({'budget': b, 'seed': s, 'n_train': len(items),
                         'macro_f1': f1, 'accuracy': acc, 'auc': auc,
                         'f1_nodefect': per[0], 'f1_defect': per[1]})
        print(f'{int(b*100):6d}% {len(items):8d}  {np.mean(f1s):8.3f}±{np.std(f1s):.3f} '
              f'{np.mean(accs):8.3f}±{np.std(accs):.3f} {np.mean(aucs):8.3f}±{np.std(aucs):.3f}',
              flush=True)

    json.dump(rows, open('results/binary_A.json', 'w'), indent=2)

    f10 = np.mean([r['macro_f1'] for r in rows if r['budget'] == 0.10])
    f100 = np.mean([r['macro_f1'] for r in rows if r['budget'] == 1.00])
    print('\n' + '=' * 74)
    print(f'GATE: binary macro-F1 at the 10% budget = {f10:.3f}   (threshold {GATE})')
    if f10 >= GATE:
        print('  => NO HEADROOM. The binary task is already solved with 10% of the real data.')
        print('     Synthetic data has nothing to add. THE BINARY TRACK IS DEAD — stopping.')
    else:
        print(f'  => HEADROOM EXISTS: {GATE-f10:.3f} macro-F1 below the gate, and '
              f'{f100-f10:.3f} below the 100% baseline ({f100:.3f}).')
        print('     Proceed: train the unconditional prior on train-pool tight crops, sample')
        print('     synthetic positives (every sample is a defect by construction), run A/B/C.')
    print(f'\nwrote results/binary_A.json')


if __name__ == '__main__':
    main()
