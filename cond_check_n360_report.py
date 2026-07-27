"""Aggregate the chunked n=360 re-check of the epoch-320 conditioning gate (audit action 1)
and report the four pre-registered statistics WITH standard errors.

The gate pass at ep320 (n=72) cleared collapse_index (0.375 vs <=0.40) and consistency_ex
(0.300 vs >=0.30) within ~1 SE of failing. This re-check pools 15 independent 24-sample
draws (4/class each) from the SAME checkpoints into one n=360 confusion matrix.

SEs: binomial for the three proportions; stratified bootstrap (resample predictions within
each conditioned-class row) for Cramer's V.
"""
import json

import numpy as np

from pcb_utils import CLASSES
from cond_check import cramers_v

TAG = 'condtight3_n360'
SRC = 'results/cond_check_ep320_n360.jsonl'
CRITERIA = {'collapse_index': ('<=', 0.40), 'consistency_ex': ('>=', 0.30),
            'cramers_v': ('>=', 0.30)}


def main():
    rows = [json.loads(l) for l in open(SRC)]
    rows = [r for r in rows if r['tag'] == TAG]
    conf = np.sum([np.array(r['confusion']) for r in rows], axis=0)
    n = int(conf.sum())

    def se_prop(p, m):
        return float(np.sqrt(p * (1 - p) / m)) if m else 0.0

    diag = int(np.trace(conf))
    consistency = diag / n
    colshare = conf.sum(0) / n
    ci = int(colshare.argmax())
    collapse = float(colshare[ci])
    keep = [i for i in range(6) if i != ci]
    n_ex = int(conf[keep].sum())
    cons_ex = sum(int(conf[i, i]) for i in keep) / n_ex
    v = cramers_v(conf.astype(float))

    # stratified bootstrap for V: conditioned counts are fixed by design, so resample each
    # row's predictions with replacement
    rng = np.random.default_rng(0)
    vs = []
    probs = conf / conf.sum(1, keepdims=True)
    ni = conf.sum(1)
    for _ in range(10000):
        b = np.stack([rng.multinomial(ni[i], probs[i]) for i in range(6)])
        vs.append(cramers_v(b.astype(float)))
    v_se = float(np.std(vs))

    stats = {'n': n, 'chunks': len(rows),
             'consistency': consistency, 'consistency_se': se_prop(consistency, n),
             'collapse_class': CLASSES[ci],
             'collapse_index': collapse, 'collapse_index_se': se_prop(collapse, n),
             'consistency_ex': cons_ex, 'consistency_ex_se': se_prop(cons_ex, n_ex),
             'n_ex': n_ex,
             'cramers_v': v, 'cramers_v_se': v_se,
             'per_class': {CLASSES[i]: float(conf[i, i] / conf[i].sum()) for i in range(6)},
             'confusion': conf.tolist()}

    print(f'=== ep320 re-check pooled over {len(rows)} chunks, n={n} (chance 16.7%) ===')
    print(f"consistency      {100*consistency:5.1f}% ± {100*stats['consistency_se']:.1f}")
    print(f"collapse_index   {collapse:.3f} ± {stats['collapse_index_se']:.3f}   "
          f"(modal class: {CLASSES[ci]})")
    print(f"consistency_ex   {100*cons_ex:5.1f}% ± {100*stats['consistency_ex_se']:.1f}   "
          f"(collapse row removed, n={n_ex})")
    print(f"cramer's V       {v:.3f} ± {v_se:.3f}")
    print('per-class: ' + '  '.join(f'{c}={100*stats["per_class"][c]:.0f}%' for c in CLASSES))
    print('confusion [row=conditioned, col=predicted]:')
    for i, c in enumerate(CLASSES):
        print(f'   {c:16s}' + ' '.join(f'{conf[i, j]:4d}' for j in range(6)))

    print('\npre-registered criteria at n=%d:' % n)
    verdicts = {}
    for k, (op, thr) in CRITERIA.items():
        val = stats[k]
        ok = val <= thr if op == '<=' else val >= thr
        verdicts[k] = bool(ok)
        print(f'   {k:16s} {val:.3f} {op} {thr}  ->  {"PASS" if ok else "FAIL"}')
    stats['criteria_pass'] = verdicts
    stats['gate_holds'] = all(verdicts.values())
    print(f'\nGATE AT n={n}: ' + ('HOLDS' if stats['gate_holds'] else 'DOES NOT HOLD'))

    with open('results/cond_check_ep320_n360_pooled.json', 'w') as fh:
        json.dump(stats, fh, indent=2)
    print('wrote results/cond_check_ep320_n360_pooled.json')


if __name__ == '__main__':
    main()
