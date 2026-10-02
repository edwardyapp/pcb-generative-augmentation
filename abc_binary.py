"""BINARY A/B/C: does synthetic data help a defect / no-defect detector under scarcity?

Synthetic samples are POSITIVES BY CONSTRUCTION -- the unconditional prior was trained only on
defect crops, so anything it draws is (meant to be) a defect. That sidesteps the class-conditioning
problem entirely: no class label is needed, so no class label can be wrong.

Conditions, per budget b and seed s:
  A  real positives + real negatives at budget b
  B  A + synthetic positives
  C  A + synthetic positives kept by the filter
FILTER (no-leakage rule): the filter at budget b, seed s IS the Condition-A model at budget b,
seed s, trained here and handed straight to the filter. A synthetic crop is kept iff that model
calls it a defect.

TWO CURVES, because the difference is the whole point:
  LEAKY   generator trained on the WHOLE train pool. At a 10% budget its samples smuggle back
          information from the other 90% -- abc_recon measured this at +0.51 macro-F1. This is
          the number the field reports.
  HONEST  generator trained ONLY on the budget-b subsample. No information the classifier does
          not already hold.
Reported side by side. If the honest curve is flat and the leaky one soars, that IS the result.

OUTCOME (added 2026-10-02): the leaky curve did not soar (+0.002 at 10%). The +0.51 cited above
is abc_recon's reconstruction ceiling, which involves no generator; it is not what a full-pool
generator delivers (AUDIT.md section 7).
"""
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_classifier import CropDS
from abc_recon import TEST_TF, filter_pool
from train_binary import pools, subsample, run, BUDGETS, SEEDS


def synth(d):
    return [(p, 1) for p in sorted(glob.glob(os.path.join(d, '*.png')))]


def main():
    device = 'cuda'
    tr, te = pools()
    vl = DataLoader(CropDS(te, TEST_TF), batch_size=64, num_workers=8)

    leaky = synth('synth_binary_pool')
    print(f'LEAKY synthetic pool (generator saw the whole train pool): {len(leaky)}')

    rows = []
    for b in BUDGETS:
        honest = synth(f'synth_binary_b{int(b*100)}') if b < 1.0 else leaky
        tag = 'honest' if b < 1.0 and honest else ('=leaky at 100%' if b >= 1.0 else 'MISSING')
        print(f'\nbudget {int(b*100)}%   honest pool: {len(honest)} ({tag})')
        for s in SEEDS:
            real = subsample(tr, b, s)
            fmodel, (f1a, acca, auca, _) = run(real, vl, device, s)   # A == the filter
            rec = {'budget': b, 'seed': s, 'n_real': len(real),
                   'A': {'macro_f1': f1a, 'accuracy': acca, 'auc': auca}}
            for name, pool in (('leaky', leaky), ('honest', honest)):
                if not pool:
                    continue
                kept = filter_pool(fmodel, pool, device)
                _, (f1b, accb, aucb, _) = run(real + pool, vl, device, s)
                _, (f1c, accc, aucc, _) = run(real + kept, vl, device, s)
                rec[f'B_{name}'] = {'macro_f1': f1b, 'accuracy': accb, 'auc': aucb,
                                    'n': len(pool)}
                rec[f'C_{name}'] = {'macro_f1': f1c, 'accuracy': accc, 'auc': aucc,
                                    'n': len(kept), 'keep_rate': len(kept) / len(pool)}
            line = f'  seed {s} | A {f1a:.3f}'
            for k in ('B_honest', 'C_honest', 'B_leaky', 'C_leaky'):
                if k in rec:
                    line += f' | {k} {rec[k]["macro_f1"]:.3f}'
            if 'C_leaky' in rec:
                line += f' | keep {100*rec["C_leaky"]["keep_rate"]:.0f}%'
            print(line, flush=True)
            rows.append(rec)

    json.dump(rows, open('results/abc_binary.json', 'w'), indent=2)

    print('\n' + '=' * 92)
    print('BINARY A/B/C  (macro-F1, mean ± std over 3 seeds)')
    print('=' * 92)
    conds = ['A', 'B_honest', 'C_honest', 'B_leaky', 'C_leaky']
    print(f'{"budget":>7} ' + ' '.join(f'{c:>14}' for c in conds))
    for b in BUDGETS:
        R = [r for r in rows if r['budget'] == b]
        line = f'{int(b*100):6d}% '
        for c in conds:
            v = np.array([r[c]['macro_f1'] for r in R if c in r])
            line += f'{v.mean():>8.3f}±{v.std():.3f} ' if v.size else f'{"—":>14} '
        a = np.mean([r['A']['macro_f1'] for r in R])
        bh = [r['B_honest']['macro_f1'] for r in R if 'B_honest' in r]
        bl = [r['B_leaky']['macro_f1'] for r in R if 'B_leaky' in r]
        if bh:
            line += f'  honest {np.mean(bh)-a:+.3f}'
        if bl:
            line += f'  leaky {np.mean(bl)-a:+.3f}'
        print(line)
    print('\nwrote results/abc_binary.json')


if __name__ == '__main__':
    main()
