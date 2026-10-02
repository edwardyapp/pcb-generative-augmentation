"""b10 A/B/C with 10 seeds and a PAIRED test. The money result, with a defensible error bar.

At 3 seeds the b10 result was A 0.419±0.011, B 0.431±0.029 -- an effect of +0.012 against a
std of 0.029. That supports "no DETECTABLE effect" and nothing stronger. With 10 seeds and a
paired test we can say something much sharper, because the design is genuinely paired: the data
subsample is FROZEN, so seed s gives A_s, B_s and C_s trained on identical data, differing only
in initialisation. The per-seed difference d_s = B_s - A_s therefore isolates the synthetic data,
with the run-to-run noise differenced away.

WHAT THE TEST CAN AND CANNOT SAY -- stated up front so the paper does not overclaim:
  CAN  : "for this budget-10 subsample, adding this synthetic pool changes macro-F1 by d ± ci."
         With a tight CI around zero that is a real null result, not merely an underpowered one.
  CANNOT: generalise over DATA subsamples. The subsample is frozen (12 generators is not
         affordable), so variance here is over classifier initialisation ONLY. A different draw
         of 215 crops could behave differently, and we do not claim otherwise.

No regeneration: the synthetic pool (synth_b10/, 360 crops) and the manifest are fixed. This is
classifier training only -- low memory, and it writes each seed to disk as it finishes, so being
killed mid-run costs nothing.
"""
import sys, pathlib  # repo layout (README.md): src/, experiments/, analysis/ importable from anywhere
sys.path[1:1] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('src', 'experiments', 'analysis')]
import argparse
import json
import os

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from pcb_utils import load_splits
from train_classifier import read_manifest, CropDS, build_train_items, build_test_items
from abc_recon import train_clf, TEST_TF, filter_pool
from abc_budget import synth_items

OUT = 'results/abc_budget_b10_seeds.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--budget', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=20)
    args = ap.parse_args()
    device = 'cuda'

    man = f'results/manifest_tight_b{args.budget}.csv'
    rows = read_manifest(man)
    tb = set(load_splits()['test_boards'])
    for r in rows:
        if r['split'] == 'train':
            assert r['board_id'] not in tb, f'LEAK: board {r["board_id"]}'

    real = build_train_items(rows, 1.0, seed=0)      # the FROZEN subsample
    test_items = build_test_items(rows)
    pool = synth_items(f'synth_b{args.budget}')
    assert pool, 'empty synthetic pool'
    vl = DataLoader(CropDS(test_items, TEST_TF), batch_size=64, shuffle=False, num_workers=8)

    print(f'budget {args.budget}%  real {len(real)} (FROZEN subsample)  synthetic {len(pool)}  '
          f'test {len(test_items)}')
    print(f'{args.seeds} seeds; variance is over CLASSIFIER INIT only (data subsample fixed)\n')

    recs = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {r['seed'] for r in recs}

    for s in range(args.seeds):
        if s in done:
            continue
        # A is trained first and IS the filter (no-leakage rule, same budget AND same seed)
        fmodel, f1a, acca, pera = train_clf(real, vl, device, s, args.epochs)
        kept = filter_pool(fmodel, pool, device)
        rec = {'seed': s, 'n_real': len(real), 'n_synth': len(pool), 'n_kept': len(kept),
               'keep_rate': len(kept) / len(pool),
               'A': {'macro_f1': f1a, 'accuracy': acca}}
        for name, items in (('B', real + pool), ('C', real + kept)):
            _, f1, acc, _ = train_clf(items, vl, device, s, args.epochs)
            rec[name] = {'macro_f1': f1, 'accuracy': acc}
        recs.append(rec)
        json.dump(recs, open(OUT, 'w'), indent=2)     # persist per seed: a kill costs nothing
        print(f'  seed {s:2d} | A {f1a:.4f} | B {rec["B"]["macro_f1"]:.4f} '
              f'(d {rec["B"]["macro_f1"]-f1a:+.4f}) | C {rec["C"]["macro_f1"]:.4f} '
              f'(d {rec["C"]["macro_f1"]-f1a:+.4f}) | keep {100*rec["keep_rate"]:.0f}%',
              flush=True)
        torch.cuda.empty_cache()

    report(recs)


def report(recs):
    recs = sorted(recs, key=lambda r: r['seed'])
    A = np.array([r['A']['macro_f1'] for r in recs])
    B = np.array([r['B']['macro_f1'] for r in recs])
    C = np.array([r['C']['macro_f1'] for r in recs])
    n = len(A)
    print('\n' + '=' * 84)
    print(f'b10 A/B/C — {n} seeds, FROZEN data subsample (variance = classifier init only)')
    print('=' * 84)
    for name, v in (('A  real only', A), ('B  + synthetic', B), ('C  + filtered', C)):
        print(f'  {name:16s} {v.mean():.4f} ± {v.std(ddof=1):.4f}   '
              f'[min {v.min():.4f}, max {v.max():.4f}]')

    print(f'\n  PAIRED comparison (each seed contributes A_s, B_s, C_s trained on identical data)')
    for name, v in (('B − A', B - A), ('C − A', C - A)):
        d = v
        m, sd = d.mean(), d.std(ddof=1)
        se = sd / np.sqrt(n)
        ci = stats.t.ppf(0.975, n - 1) * se
        t, p_t = stats.ttest_rel(B if name.startswith('B') else C, A)
        try:
            w, p_w = stats.wilcoxon(d)
        except Exception:
            p_w = float('nan')
        coh = m / sd if sd > 0 else 0.0
        print(f'\n  {name}:  mean {m:+.4f}   95% CI [{m-ci:+.4f}, {m+ci:+.4f}]')
        print(f'          paired t({n-1}) = {t:+.2f}, p = {p_t:.4f}   '
              f'Wilcoxon p = {p_w:.4f}   Cohen d = {coh:+.2f}')
        sig = p_t < 0.05
        if sig and m > 0:
            print(f'          => synthetic data HELPS (significant, but by {m:+.4f})')
        elif sig and m < 0:
            print(f'          => synthetic data HURTS (significant)')
        else:
            print(f'          => NO significant effect. The CI brackets zero.')

    # how big an effect could we have missed? (minimum detectable effect at 80% power)
    sd = (B - A).std(ddof=1)
    mde = 2.8 * sd / np.sqrt(n)          # ~t(.975)+t(.80) for modest n
    print(f'\n  POWER: with {n} seeds and a paired sd of {sd:.4f}, the smallest effect we could')
    print(f'  reliably detect is about {mde:+.4f} macro-F1. Anything smaller than that we cannot')
    print(f'  rule out — so the honest claim is "no effect larger than ~{mde:.3f}".')
    print(f'\n  For scale, at this same budget (3-seed runs, each against its own A):')
    for label, path, key in (('full-pool generator  ', 'results/abc_leaky_b10.json', 'B'),
                             ('reconstruction ceiling', 'results/abc_recon.json', 'B_pool')):
        if not os.path.exists(path):
            continue
        rs = [r for r in json.load(open(path)) if abs(r['budget'] - 0.10) < 1e-9 or r['budget'] == 10]
        d = np.mean([r[key]['macro_f1'] - r['A']['macro_f1'] for r in rs])
        print(f'    {label.strip():22s} {d:+.3f}  ({abs(d)/max(abs(mde),1e-9):.1f}x the detection floor)')
    print(f'  The ceiling uses no generator (VQ-VAE round-trips of all 2,149 real crops); it bounds')
    print(f'  what images of this form could carry, not what a leaky generator delivers.')
    print(f'\n  filter keep-rate {100*np.mean([r["keep_rate"] for r in recs]):.1f}%')
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
