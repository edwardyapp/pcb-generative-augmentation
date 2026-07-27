"""The money figure: the HONEST curve (budget-restricted generator) beside the LEAKY one.

Three curves, and the distance between them is the paper's central claim:

  A        real data only.
  B/C hon  real + samples from a generator that saw ONLY the budget-b subsample. No leakage
           anywhere: VQ-VAE, top prior and bottom prior all trained from scratch on exactly
           the crops the classifier holds.
  B_pool   real + reconstructions of the FULL train pool. This is what you get when the
           generator is trained on all the real data and then "evaluated" at a reduced budget
           -- the standard setup. At a 10% budget it scores 0.910 against a full-data ceiling
           of 0.918: it is not generation, it is the training set coming back.

If the honest curve sits near A while the leaky curve flies, the field's headline result is
an artefact of the protocol, not a property of generative models.
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BUDGETS = [10, 25, 50, 100]
RECON = 'results/abc_recon.json'


def honest(b):
    p = f'results/abc_budget_b{b}.json'
    if not os.path.exists(p):
        return None
    rs = json.load(open(p))
    return {c: (np.mean([r[c]['macro_f1'] for r in rs]),
                np.std([r[c]['macro_f1'] for r in rs])) for c in ('A', 'B', 'C')} | \
           {'keep': np.mean([r['keep_rate'] for r in rs])}


def leaky():
    if not os.path.exists(RECON):
        return {}
    R = json.load(open(RECON))
    out = {}
    for b in BUDGETS:
        f = lambda c: np.array([r[c]['macro_f1'] for r in R if abs(r['budget'] - b / 100) < 1e-9])
        out[b] = {c: (f(c).mean(), f(c).std()) for c in ('A', 'B_pool', 'B_self')}
    return out


def main():
    H = {b: honest(b) for b in BUDGETS}
    L = leaky()
    have = [b for b in BUDGETS if H[b]]
    if not have:
        print('no budget-restricted results yet (gate not passed / pipeline not run)')
        print('the leaky curve is available; the honest curve will appear here when it lands')

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    if L:
        x = BUDGETS
        a = [L[b]['A'][0] for b in x]
        ae = [L[b]['A'][1] for b in x]
        ax.errorbar(x, a, yerr=ae, marker='o', lw=2.4, capsize=4, color='#444444',
                    label='A — real data only')
        bp = [L[b]['B_pool'][0] for b in x]
        ax.errorbar(x, bp, yerr=[L[b]['B_pool'][1] for b in x], marker='s', lw=2.4, capsize=4,
                    ls='--', color='#C0473B',
                    label='B_pool — generator trained on the FULL pool (LEAKY)')
        ax.axhline(L[100]['A'][0], ls=':', c='#888', lw=1.3)
        ax.text(101, L[100]['A'][0] + .012, f'full-data ceiling {L[100]["A"][0]:.3f}',
                ha='right', fontsize=8, color='#666')
    if have:
        hb = [H[b]['B'][0] for b in have]
        hc = [H[b]['C'][0] for b in have]
        ax.errorbar(have, hb, yerr=[H[b]['B'][1] for b in have], marker='D', lw=2.4, capsize=4,
                    color='#2C7FB8', label='B — budget-restricted generator (HONEST)')
        ax.errorbar(have, hc, yerr=[H[b]['C'][1] for b in have], marker='v', lw=1.8, capsize=3,
                    ls='--', color='#7FC6E8', label='C — + quality filter (HONEST)')
        for b in have:
            d = H[b]['B'][0] - H[b]['A'][0]
            ax.annotate(f'{d:+.3f}', (b, H[b]['B'][0]), textcoords='offset points',
                        xytext=(0, 9), ha='center', fontsize=8, color='#2C7FB8')

    ax.axhline(1 / 6, ls='--', c='gray', lw=1)
    ax.text(100, 1 / 6 + .012, 'chance', ha='right', color='gray', fontsize=8)
    ax.set_xlabel('real-data budget (% of train pool)')
    ax.set_ylabel('macro-F1 (test boards 06+09)')
    ax.set_title('Does synthetic data cure data scarcity?\n'
                 'The honest answer and the leaky one, on the same axes')
    ax.set_xticks(BUDGETS); ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3)
    fig.text(0.01, -0.03, 'Variance = classifier initialisation only; the data subsample is '
                          'frozen per budget (12 generators is not affordable).',
             fontsize=7.5, color='#666')
    plt.tight_layout()
    plt.savefig('figures/fig11_budget_abc.png', bbox_inches='tight', dpi=120)
    print('wrote figures/fig11_budget_abc.png')

    if have:
        print(f'\n{"budget":>7} {"A":>16} {"B honest":>16} {"C honest":>16} {"B_pool leaky":>16} {"keep":>6}')
        for b in have:
            h, l = H[b], L.get(b, {})
            lp = f'{l["B_pool"][0]:.3f}±{l["B_pool"][1]:.3f}' if l else '-'
            print(f'{b:>6}% {h["A"][0]:>9.3f}±{h["A"][1]:.3f} {h["B"][0]:>9.3f}±{h["B"][1]:.3f} '
                  f'{h["C"][0]:>9.3f}±{h["C"][1]:.3f} {lp:>16} {100*h["keep"]:>5.0f}%')


if __name__ == '__main__':
    main()
