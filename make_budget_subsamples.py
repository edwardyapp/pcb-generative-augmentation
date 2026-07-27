"""Freeze ONE training subsample per budget, and write it as a self-contained manifest.

This is the anti-leakage keystone of the budget-restricted protocol. At budget b, the
generator (VQ-VAE + both priors) and the classifier (A, B, C) must be trained on EXACTLY
the same crops. If the generator sees even one crop the budget-b classifier is not allowed
to have, the whole experiment leaks across the subsample boundary and reproduces the very
artefact we are trying to expose (see abc_recon: B_pool @10% = 0.910 ~ the full-data 0.918).

The subsample is produced by train_classifier.build_train_items(rows, b, seed=0) -- the
SAME function the classifier uses -- so the two cannot drift apart. Each manifest then
contains those train rows plus the FULL, untouched test set, so downstream code can be run
with `--budget 1.0 --manifest results/manifest_tight_b10.csv` and get exactly the subsample.

Consequence for variance, stated plainly here and in the paper: with the subsample frozen,
the 3 seeds vary CLASSIFIER INITIALISATION ONLY, not data subsampling. Retraining 12
generators (4 budgets x 3 data draws) is not affordable, so we do not pretend to have done it.
"""
import csv
import json
import os
from collections import Counter

from pcb_utils import CLASSES, load_splits
from train_classifier import read_manifest, build_train_items

SRC = 'results/manifest_tight.csv'      # size-matched set: one primary crop per base image
BUDGETS = [0.10, 0.25, 0.50]            # 100% is the whole train pool = the in-flight run
DATA_SEED = 0                           # the ONE data draw; frozen for every budget
FIELDS = ['crop_path', 'board_id', 'class_name', 'class_idx', 'variant', 'split',
          'is_primary', 'base_crop']


def main():
    rows = read_manifest(SRC)
    test_boards = set(load_splits()['test_boards'])
    test_rows = [r for r in rows if r['split'] == 'test']
    by_path = {r['crop_path']: r for r in rows}

    prov = {'src': SRC, 'data_seed': DATA_SEED, 'test_boards': sorted(test_boards),
            'note': 'generator and classifier at budget b train on exactly these crops',
            'budgets': {}}

    for b in BUDGETS:
        items = build_train_items(rows, b, DATA_SEED)          # the SAME call the classifier makes
        train_rows = [by_path[p] for p, _ in items]

        for r in train_rows:                                   # hard no-leakage assertions
            assert r['split'] == 'train', f'non-train row in subsample: {r["crop_path"]}'
            assert r['board_id'] not in test_boards, f'LEAK: test board {r["board_id"]}'
        paths = [r['crop_path'] for r in train_rows]
        assert len(set(paths)) == len(paths), 'duplicate crop in subsample'

        out = f'results/manifest_tight_b{int(b*100)}.csv'
        with open(out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(train_rows + test_rows)

        cnt = Counter(r['class_name'] for r in train_rows)
        prov['budgets'][f'{int(b*100)}'] = {
            'manifest': out, 'n_train': len(train_rows), 'n_test': len(test_rows),
            'per_class': {c: cnt[c] for c in CLASSES}, 'crops': sorted(paths)}
        print(f'budget {int(b*100):3d}%  train {len(train_rows):4d}  test {len(test_rows)}  '
              + ' '.join(f'{c[:4]}={cnt[c]}' for c in CLASSES) + f'  -> {out}')

    # nesting check: a smaller budget must be a SUBSET of a larger one (same seed, same shuffle)
    sets = {k: set(v['crops']) for k, v in prov['budgets'].items()}
    for a, bb in [('10', '25'), ('25', '50')]:
        ok = sets[a] <= sets[bb]
        prov['budgets'][a][f'subset_of_{bb}'] = ok
        print(f'  {a}% is a subset of {bb}%: {ok}')

    with open('results/budget_subsamples.json', 'w') as fh:
        json.dump(prov, fh, indent=2)
    print('\nwrote results/budget_subsamples.json (full crop list per budget, for provenance)')


if __name__ == '__main__':
    main()
