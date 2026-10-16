"""Stage 5/6 classifier: ResNet-18 (ImageNet-pretrained), 6-way defect classifier.

Board-split evaluation on held-out test boards. Doubles as:
  - Condition A baseline (real only)
  - the Stage-5 quality filter at the same budget (no-leakage rule)

Real-data unit = base (plain) crop, stratified subsample by class at --budget.
Augmentation (flip/rotation) applied on the fly, identically across all conditions.
"""
import argparse
import csv
import json
import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from pcb_utils import CLASSES, load_splits


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def read_manifest(path='results/manifest.csv'):
    with open(path) as fh:
        return list(csv.DictReader(fh))


class CropDS(Dataset):
    def __init__(self, items, tf):
        self.items = items
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tf(Image.open(path).convert('RGB')), label


def build_train_items(rows, budget, seed):
    byc = defaultdict(list)
    for r in rows:
        if r['split'] == 'train' and r['variant'] == 'plain':
            byc[r['class_name']].append((r['crop_path'], int(r['class_idx'])))
    rng = random.Random(seed)
    items = []
    for c in CLASSES:
        lst = sorted(byc[c])
        rng.shuffle(lst)
        k = len(lst) if budget >= 1.0 else max(1, int(round(len(lst) * budget)))
        items += lst[:k]
    return items


def build_test_items(rows):
    return [(r['crop_path'], int(r['class_idx'])) for r in rows
            if r['split'] == 'test' and r['variant'] == 'plain']


def macro_f1(y_true, y_pred, n=6):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    f1s, per = [], {}
    for c in range(n):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        per[CLASSES[c]] = round(f1, 4)
    return float(np.mean(f1s)), float((y_true == y_pred).mean()), per


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    yt, yp = [], []
    for x, y in loader:
        p = model(x.to(device)).argmax(1).cpu().numpy()
        yp += p.tolist()
        yt += y.numpy().tolist()
    return macro_f1(yt, yp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--condition', default='A')
    ap.add_argument('--manifest', default='results/manifest.csv')
    ap.add_argument('--out', default=None)
    ap.add_argument('--save_model', default=None)
    args = ap.parse_args()
    device = 'cuda'
    set_seed(args.seed)

    splits = load_splits()
    rows = read_manifest(args.manifest)
    test_boards = set(splits['test_boards'])
    for r in rows:  # no-leakage assertion
        if r['split'] == 'train':
            assert r['board_id'] not in test_boards, f"LEAK board {r['board_id']}"

    train_items = build_train_items(rows, args.budget, args.seed)
    test_items = build_test_items(rows)

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(), norm,
    ])
    test_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])

    tl = DataLoader(CropDS(train_items, train_tf), batch_size=args.batch, shuffle=True,
                    num_workers=8, drop_last=False)
    vl = DataLoader(CropDS(test_items, test_tf), batch_size=args.batch, shuffle=False, num_workers=8)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 6)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    print(f'[cond {args.condition} budget {args.budget} seed {args.seed}] '
          f'train={len(train_items)} test={len(test_items)} '
          f'test_boards={splits["test_boards"]} chance_macroF1~{1/6:.3f}', flush=True)

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for x, y in tl:
            opt.zero_grad()
            loss = crit(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
        f1, acc, per = evaluate(model, vl, device)
        best = max(best, f1)
        print(f'epoch {ep+1}/{args.epochs} test_macroF1={f1:.4f} acc={acc:.4f} best={best:.4f}', flush=True)

    f1, acc, per = evaluate(model, vl, device)
    print(f'RESULT cond={args.condition} budget={args.budget} seed={args.seed} '
          f'macro_f1={f1:.4f} accuracy={acc:.4f} best_macro_f1={best:.4f}', flush=True)
    print('per_class_f1:', per, flush=True)

    res = {'condition': args.condition, 'budget': args.budget, 'seed': args.seed,
           'macro_f1': f1, 'accuracy': acc, 'best_macro_f1': best, 'per_class_f1': per,
           'n_train': len(train_items), 'n_test': len(test_items),
           'test_boards': splits['test_boards']}
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(res, fh, indent=2)
    if args.save_model:
        torch.save(model.state_dict(), args.save_model)


if __name__ == '__main__':
    main()
