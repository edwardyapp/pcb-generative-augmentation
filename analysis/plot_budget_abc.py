"""Fig 11: what synthetic data adds over real data alone, for two generators, at each budget.

Two arms, each measured against ITS OWN Condition-A control (same frozen subsample, same seeds,
trained in the same run):

  honest     results/abc_budget_b{b}.json  generator (VQ-VAE + both priors) trained from scratch
                                           on exactly the budget-b subsample the classifier holds.
  full pool  results/abc_leaky_b{b}.json   generator trained on the FULL train pool (prior_b100,
                                           synth_b100/); the classifier still sees only budget b.
                                           At 100% the two arms are the same pool, so it is
                                           plotted only at 10/25/50.

Both pools hold 360 generated samples and pass through the same no-leakage filter (the budget-b
Condition-A model). Lift = mean over seeds of the per-seed difference B_s - A_s (or C_s - A_s);
error bars are the sd of those paired differences.

What this figure is NOT: abc_recon.json's B_pool. That arm has no generator at all -- it adds
VQ-VAE reconstructions of all 2,149 real train crops -- and its +0.510 at 10% is a
reconstruction ceiling, not a leaky-generator result (AUDIT.md section 7). It is plotted in
fig10 (plot_abc.py), never here.
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BUDGETS = [10, 25, 50, 100]
NOISE = 0.015          # run-to-run band on identical data and seeds, 6-way (AUDIT.md 7.5)
ARMS = [('honest', 'results/abc_budget_b{}.json', 'budget-restricted generator', '#2a78d6', 'o', -1),
        ('full', 'results/abc_leaky_b{}.json', 'full-pool generator', '#eb6834', 's', 1)]


def load(pattern, b):
    p = pattern.format(b)
    if not os.path.exists(p):
        return None
    rs = sorted(json.load(open(p)), key=lambda r: r['seed'])
    f = lambda c: np.array([r[c]['macro_f1'] for r in rs])
    A = f('A')
    return {'A': A.mean(), 'n': len(rs),
            'B': ((f('B') - A).mean(), (f('B') - A).std()),
            'C': ((f('C') - A).mean(), (f('C') - A).std()),
            'keep': np.mean([r['keep_rate'] for r in rs])}


def main():
    R = {key: {b: load(pat, b) for b in BUDGETS} for key, pat, *_ in ARMS}
    if not any(R['honest'].values()):
        print('no budget-restricted results yet (results/abc_budget_b*.json)')
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, cond, title in ((axes[0], 'B', '(a) B: + all 360 samples'),
                            (axes[1], 'C', '(b) C: + filtered samples')):
        ax.axhspan(-NOISE, NOISE, color='#e8e8e4', zorder=0)
        ax.axhline(0, color='#888', lw=1, zorder=1)
        for key, _, label, col, mk, dx in ARMS:
            have = [b for b in BUDGETS if R[key][b]]
            m = [R[key][b][cond][0] for b in have]
            s = [R[key][b][cond][1] for b in have]
            ax.errorbar([b + dx for b in have], m, yerr=s, marker=mk, ms=7, lw=2, capsize=3, color=col,
                        mec='white', mew=1.2, label=label, zorder=3)
        ax.set_title(title, fontsize=10, loc='left')
        ax.set_xticks(BUDGETS)
        ax.set_xlabel('real-data budget (% of train pool)')
        ax.grid(alpha=0.3, axis='y')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel('macro-F1 minus own Condition A')
    axes[0].text(100, NOISE + 0.003, f'±{NOISE} rerun noise', ha='right', va='bottom',
                 fontsize=8, color='#666')
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=8.5, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.95))
    fig.suptitle('No gain outside rerun noise from either generator; '
                 'unfiltered samples hurt above 10%', fontsize=10.5)
    fig.text(0.01, -0.04,
             'Each arm is differenced against its own Condition-A run (same frozen subsample, '
             'same seeds); bars = sd of the paired differences over seeds. At 100% the arms are '
             'the same pool.\nNot shown: the reconstruction ceiling (VQ-VAE round-trip of all '
             '2,149 real crops, no generator), +0.510 at 10% -- see fig10.',
             fontsize=7.5, color='#666')
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig('figures/fig11_budget_abc.png', bbox_inches='tight', dpi=120)
    print('wrote figures/fig11_budget_abc.png')

    print(f'\n{"budget":>7} {"A honest":>9} {"B-A":>8} {"C-A":>8} {"keep":>6}   '
          f'{"A full":>9} {"B-A":>8} {"C-A":>8} {"keep":>6}')
    for b in BUDGETS:
        h, l = R['honest'][b], R['full'][b]
        if not h:
            continue
        row = (f'{b:>6}% {h["A"]:>9.3f} {h["B"][0]:>+8.3f} {h["C"][0]:>+8.3f} '
               f'{100*h["keep"]:>5.0f}%   ')
        row += (f'{l["A"]:>9.3f} {l["B"][0]:>+8.3f} {l["C"][0]:>+8.3f} {100*l["keep"]:>5.0f}%'
                if l else f'{"(same pool as honest)":>35}')
        print(row)


if __name__ == '__main__':
    main()
