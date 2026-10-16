"""Plot conditioning consistency vs training epoch, plus the PRE-REGISTERED convergence test.

The point of this figure is that it replaces a single pass/fail with a trend:
  * if consistency CLIMBS with epochs, the epoch-80 negative result was premature and more
    training is the answer;
  * if it is FLAT at chance across every checkpoint, the negative result is robust, and we
    can say so having actually looked.

A pooled consistency number is misleading under collapse, so three curves are drawn, not one:
  consistency, consistency-excluding-the-collapse-class, and Cramer's V (which cannot be
  inflated by the model dumping every sample into a single class).

CONVERGENCE CRITERION — fixed in advance, at the user's instruction:
    converged  <=>  |relative fall in loss over the last 20 epochs| < 0.5%
                AND bottom-prior code accuracy has plateaued (relative gain < 0.5% over 20 ep)
If it is still descending at epoch 320, we say so rather than declare convergence.
"""
import sys, pathlib  # repo layout (README.md): src/, experiments/, analysis/ importable from anywhere
sys.path[1:1] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('src', 'experiments', 'analysis')]
import os
import re
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pcb_utils import CLASSES
from plot_loss_curves import parse, curve

CHECKS = 'results/cond_checks.jsonl'
LOGS = ['logs/rebase_tight2_5090.log', 'logs/converge_5090.log']
CHANCE = 1 / 6
THRESH = 0.005          # 0.5%, the pre-registered convergence threshold


def load_checks():
    if not os.path.exists(CHECKS):
        return []
    rs = [json.loads(l) for l in open(CHECKS) if l.strip()]
    return sorted(rs, key=lambda r: r['epoch'])


def merged_curves():
    """Stitch condtight2 (abs 1-80) and condtight3 (local 1-240 => abs 81-320)."""
    out = {}
    for hier in ('top', 'bottom'):
        eps, loss, acc = [], [], []
        d2 = parse('logs/rebase_tight2_5090.log')
        e, l, a = curve(d2, hier)
        eps += list(e); loss += list(l); acc += list(a)
        d3 = parse('logs/converge_5090.log')
        e, l, a = curve(d3, hier)
        eps += [x + 80 for x in e]; loss += list(l); acc += list(a)
        o = np.argsort(eps)
        out[hier] = (np.array(eps)[o], np.array(loss)[o], np.array(acc)[o])
    return out


def convergence(eps, vals, at, tail=20, rising=False):
    """Relative change over the `tail` epochs ending at `at`. Converged if < THRESH."""
    m = (eps > at - tail) & (eps <= at)
    if m.sum() < max(3, tail // 2):
        return None
    v = vals[m]
    rel = (v[0] - v[-1]) / abs(v[0]) if not rising else (v[-1] - v[0]) / max(abs(v[0]), 1e-9)
    return {'rel': rel, 'converged': bool(rel < THRESH), 'first': float(v[0]), 'last': float(v[-1])}


def main():
    rs = load_checks()
    cur = merged_curves()

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))

    # ---- (a) consistency vs epoch ----
    if rs:
        e = [r['epoch'] for r in rs]
        ax[0].plot(e, [100 * r['consistency'] for r in rs], 'o-', lw=2.2, color='#2C7FB8',
                   label='consistency (pooled)')
        ax[0].plot(e, [100 * r['consistency_ex'] for r in rs], 's--', lw=2, color='#C0473B',
                   label='excluding collapse class')
        for r in rs:
            if r['collapsed']:
                ax[0].annotate(f"collapse:\n{r['collapse_class']}\n{100*r['collapse_index']:.0f}%",
                               (r['epoch'], 100 * r['consistency']), textcoords='offset points',
                               xytext=(0, 12), ha='center', fontsize=7, color='#C0473B')
    ax[0].axhline(100 * CHANCE, ls='--', c='gray', lw=1.2)
    ax[0].text(0.99, 100 * CHANCE + 0.6, 'chance 16.7%', transform=ax[0].get_yaxis_transform(),
               ha='right', color='gray', fontsize=8)
    ax[0].set_xlabel('prior training epoch (absolute)')
    ax[0].set_ylabel('conditioning consistency (%)')
    ax[0].set_title('(a) Does conditioning improve with training?')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3); ax[0].set_ylim(0, 60)

    # ---- (b) Cramer's V — collapse-proof ----
    if rs:
        ax[1].plot([r['epoch'] for r in rs], [r['cramers_v'] for r in rs], 'o-', lw=2.2,
                   color='#3B9C5A')
    ax[1].axhline(0, ls='--', c='gray', lw=1.2)
    ax[1].set_xlabel('prior training epoch (absolute)')
    ax[1].set_ylabel("Cramer's V (conditioned vs predicted)")
    ax[1].set_title("(b) Association between label and output\n0 = the class label does nothing")
    ax[1].grid(alpha=0.3)

    # ---- (c) loss, with the convergence window ----
    for hier, col in (('top', '#2C7FB8'), ('bottom', '#C0473B')):
        eps, loss, _ = cur[hier]
        if len(eps):
            ax[2].plot(eps, loss, lw=2, color=col, label=f'{hier} loss')
    ax[2].set_xlabel('prior training epoch (absolute)')
    ax[2].set_ylabel('training loss')
    ax[2].set_title('(c) Prior loss (condtight2 + condtight3)')
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/fig9_consistency_vs_epoch.png', bbox_inches='tight', dpi=120)
    plt.close()

    # ---- report ----
    print('=' * 100)
    print('CONDITIONING vs EPOCH   (chance 16.7%)')
    print('=' * 100)
    print(f'{"epoch":>6} {"consist":>8} {"ex-collapse":>12} {"cramersV":>9} {"collapse":>22}  per-class')
    for r in rs:
        pc = ' '.join(f'{c[:4]}={100*r["per_class"][c]:.0f}' for c in CLASSES)
        flag = f'{r["collapse_class"]} {100*r["collapse_index"]:.0f}%' + (' **' if r['collapsed'] else '')
        print(f'{r["epoch"]:>6} {100*r["consistency"]:>7.1f}% {100*r["consistency_ex"]:>11.1f}% '
              f'{r["cramers_v"]:>9.3f} {flag:>22}  {pc}')

    print('\n' + '=' * 100)
    print(f'CONVERGENCE (pre-registered: relative change over last 20 epochs < {100*THRESH}%)')
    print('=' * 100)
    for r in rs:
        at = r['epoch']
        parts = []
        for hier in ('top', 'bottom'):
            eps, loss, acc = cur[hier]
            cl = convergence(eps, loss, at)
            parts.append(f'{hier} loss ' + (f'fell {100*cl["rel"]:.2f}% over 20ep -> '
                         f'{"CONVERGED" if cl["converged"] else "still falling"}'
                         if cl else 'n/a'))
        eps, loss, acc = cur['bottom']
        ca = convergence(eps, acc, at, rising=True)
        parts.append('bottom acc ' + (f'{100*ca["rel"]:+.2f}%/20ep '
                     f'{"plateaued" if ca["converged"] else "still climbing"}'
                     if ca else 'n/a'))
        print(f'  epoch {at:>3}: ' + ' | '.join(parts))

    print('\nwrote figures/fig9_consistency_vs_epoch.png')


if __name__ == '__main__':
    main()
