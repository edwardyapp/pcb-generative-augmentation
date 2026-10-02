"""A/B/C at one budget, with a BUDGET-RESTRICTED generator. The honest experiment.

Everything the generator saw at budget b is exactly what the classifier gets at budget b:
the frozen subsample in results/manifest_tight_b{b}.csv. The VQ-VAE and both priors were
trained from scratch on it -- no warm-start from any full-pool checkpoint, because a
full-pool DECODER has already learned to render defects from crops this budget is not
allowed to have seen.

Conditions:
  A  real subsample only                       (also serves as the filter, per the rule below)
  B  real + the full synthetic pool
  C  real + synthetic kept by the filter

NO-LEAKAGE RULE, enforced by construction: the filter at budget b, seed s IS the Condition-A
model at budget b, seed s -- trained here, in this process, and handed straight to the filter.
A synthetic crop is kept iff that model predicts the class the prior was conditioned on.

VARIANCE, stated plainly: the data subsample is FROZEN per budget. The 3 seeds vary
CLASSIFIER INITIALISATION ONLY, not data subsampling. Retraining 12 generators (4 budgets x
3 data draws) is not affordable, and we do not pretend otherwise. The error bars below are
therefore narrower than a full data-resampling study would give.
"""
import sys, pathlib  # repo layout (README.md): src/, experiments/, analysis/ importable from anywhere
sys.path[1:1] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('src', 'experiments', 'analysis')]
import os
import json
import glob
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image

from pcb_utils import CLASSES, CLASS_TO_IDX, load_splits
from train_classifier import read_manifest, CropDS, build_train_items, build_test_items, evaluate
from abc_recon import train_clf, TRAIN_TF, TEST_TF, filter_pool

SEEDS = [0, 1, 2]


def synth_items(d):
    """Pool crops are named {class}_{idx}.png; the class IS the conditioned label."""
    out = []
    for p in sorted(glob.glob(os.path.join(d, '*.png'))):
        cls = os.path.basename(p).rsplit('_', 1)[0]
        assert cls in CLASS_TO_IDX, f'unparseable synthetic crop: {p}'
        out.append((p, CLASS_TO_IDX[cls]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, required=True)      # 10 / 25 / 50 / 100
    ap.add_argument('--synth', required=True)                 # dir of labelled synthetic crops
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    device = 'cuda'

    man = args.manifest or (f'results/manifest_tight_b{args.budget}.csv' if args.budget < 100
                            else 'results/manifest_tight.csv')
    out = args.out or f'results/abc_budget_b{args.budget}.json'

    rows = read_manifest(man)
    test_boards = set(load_splits()['test_boards'])
    for r in rows:
        if r['split'] == 'train':
            assert r['board_id'] not in test_boards, f'LEAK: board {r["board_id"]}'

    # budget 1.0 on the RESTRICTED manifest == the frozen subsample, identical for every seed
    real = build_train_items(rows, 1.0, seed=0)
    test_items = build_test_items(rows)
    pool = synth_items(args.synth)

    # This ran once with an EMPTY pool and produced B and C numbers that were simply A retrained
    # -- plausible-looking, entirely meaningless. A silent zero is the most dangerous input there
    # is, so refuse it loudly.
    assert pool, (f'synthetic pool {args.synth} is EMPTY -- the generator did not produce '
                  f'samples. Refusing to fabricate B/C numbers that are just A.')
    assert len(pool) >= 60, f'synthetic pool {args.synth} has only {len(pool)} crops; expected 360'
    vl = DataLoader(CropDS(test_items, TEST_TF), batch_size=64, shuffle=False, num_workers=8)

    print(f'budget {args.budget}%  real {len(real)}  synthetic {len(pool)}  test {len(test_items)}')
    print(f'  manifest {man}\n  synth    {args.synth}')
    print('  variance = classifier init only (data subsample frozen)\n')

    recs = []
    for s in SEEDS:
        # --- A: real only. This model IS the filter (no-leakage rule). ---
        fmodel, f1a, acca, pera = train_clf(real, vl, device, s, args.epochs)
        kept = filter_pool(fmodel, pool, device)

        rec = {'budget': args.budget, 'seed': s, 'n_real': len(real), 'n_synth': len(pool),
               'n_kept': len(kept), 'keep_rate': len(kept) / max(1, len(pool)),
               'A': {'macro_f1': f1a, 'accuracy': acca, 'per_class_f1': pera}}
        for name, items in (('B', real + pool), ('C', real + kept)):
            _, f1, acc, per = train_clf(items, vl, device, s, args.epochs)
            rec[name] = {'macro_f1': f1, 'accuracy': acc, 'per_class_f1': per,
                         'n_train': len(items)}
        print(f'  seed {s} | A {f1a:.3f} | B {rec["B"]["macro_f1"]:.3f} | '
              f'C {rec["C"]["macro_f1"]:.3f} | filter kept {100*rec["keep_rate"]:.0f}%', flush=True)
        recs.append(rec)

    with open(out, 'w') as fh:
        json.dump(recs, fh, indent=2)

    print(f'\nbudget {args.budget}%  (mean +/- std over 3 classifier inits)')
    for c in ('A', 'B', 'C'):
        v = np.array([r[c]['macro_f1'] for r in recs])
        d = v.mean() - np.mean([r['A']['macro_f1'] for r in recs])
        print(f'  {c}: {v.mean():.3f} +/- {v.std():.3f}' + (f'   ({d:+.3f} vs A)' if c != 'A' else ''))
    print(f'  filter keep-rate: {100*np.mean([r["keep_rate"] for r in recs]):.1f}%')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
