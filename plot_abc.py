"""Fig 10: the A/B/C reconstruction CEILING -- what images of this form could deliver.

No generator is involved anywhere in this figure: abc_recon.py never imports PixelSNAIL. The
"synthetic" crops are VQ-VAE round-trips of real crops, either of the classifier's own budget
(B_self/C_self) or of all 2,149 train-pool crops (B_pool/C_pool). B_pool at a 10% budget
(0.910) recovers the full-data baseline (0.918): +0.510 is an upper bound on what a generator
that reproduced its training set faithfully could buy.

It is NOT what a leaky (full-pool) generator delivers. Run as an actual generator experiment
(abc_leaky_b*.json, AUDIT.md section 7) that arm gives +0.002 at 10% -- see fig11. Earlier
versions of this figure labelled B_pool "LEAKY" and its gap "leakage"; that framing is
retracted.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = json.load(open('results/abc_recon.json'))
BUDGETS = [0.10, 0.25, 0.50, 1.00]
CONDS = [('A', 'A — real only', '#444444', '-'),
         ('B_self', 'B — + recon of YOUR OWN budget', '#2C7FB8', '-'),
         ('C_self', 'C — + filtered recon, own budget', '#7FC6E8', '--'),
         ('B_pool', 'B — + recon of FULL train pool (ceiling)', '#C0473B', '-'),
         ('C_pool', 'C — + filtered recon, full pool (ceiling)', '#E89A93', '--')]


def stat(c, b):
    v = np.array([r[c]['macro_f1'] for r in R if r['budget'] == b])
    return v.mean(), v.std()


fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
x = [b * 100 for b in BUDGETS]
for key, lab, col, ls in CONDS:
    m = [stat(key, b)[0] for b in BUDGETS]
    s = [stat(key, b)[1] for b in BUDGETS]
    ax[0].errorbar(x, m, yerr=s, marker='o', capsize=4, lw=2.2, ls=ls, color=col, label=lab)

full = stat('A', 1.00)[0]
ax[0].axhline(full, ls=':', c='#C0473B', lw=1.5)
ax[0].text(101, full + 0.012, f'A @100% = {full:.3f}\n(the full real dataset)',
           color='#C0473B', fontsize=8, ha='right')
ax[0].axhline(1 / 6, ls='--', c='gray', lw=1)
ax[0].text(100, 1 / 6 + 0.012, 'chance', color='gray', ha='right', fontsize=8)
ax[0].annotate('', xy=(10, stat('B_pool', .10)[0]), xytext=(10, stat('A', .10)[0]),
               arrowprops=dict(arrowstyle='<->', color='#C0473B', lw=1.6))
ax[0].text(12, 0.66, 'recon ceiling\n+0.51', color='#C0473B', fontsize=9, fontweight='bold')
ax[0].annotate('', xy=(10, stat('C_self', .10)[0]), xytext=(10, stat('A', .10)[0]),
               arrowprops=dict(arrowstyle='<->', color='#2C7FB8', lw=1.6))
ax[0].text(4.5, 0.44, 'own budget\n+0.09', color='#2C7FB8', fontsize=9, fontweight='bold')
ax[0].set_xlabel('real-data budget (% of train pool)')
ax[0].set_ylabel('macro-F1 (test boards 06+09)')
ax[0].set_title('(a) Reconstructions as "synthetic": the upper bound (no generator)\n'
                'B_pool @10% recovers the full-data baseline; a full-pool generator gives +0.002')
ax[0].set_xticks([10, 25, 50, 100]); ax[0].set_ylim(0, 1.0)
ax[0].legend(fontsize=8, loc='lower right'); ax[0].grid(alpha=0.3)

# right: lift over A, honest vs leaky
w = 0.35
idx = np.arange(4)
honest = [max(stat('B_self', b)[0], stat('C_self', b)[0]) - stat('A', b)[0] for b in BUDGETS]
leaky = [stat('B_pool', b)[0] - stat('A', b)[0] for b in BUDGETS]
ax[1].bar(idx - w / 2, leaky, w, color='#C0473B', label='recon of full train pool (ceiling)')
ax[1].bar(idx + w / 2, honest, w, color='#2C7FB8', label='recon of your own budget')
for i, (l, h) in enumerate(zip(leaky, honest)):
    ax[1].text(i - w / 2, l + .012, f'{l:+.2f}', ha='center', fontsize=9, color='#C0473B')
    ax[1].text(i + w / 2, h + .012, f'{h:+.2f}', ha='center', fontsize=9, color='#2C7FB8')
ax[1].axhline(0, c='k', lw=1)
ax[1].set_xticks(idx); ax[1].set_xticklabels([f'{int(b*100)}%' for b in BUDGETS])
ax[1].set_xlabel('real-data budget')
ax[1].set_ylabel('macro-F1 lift over Condition A')
ax[1].set_title('(b) Lift over A from reconstructions, not generated samples\n'
                '(generated-sample lift: fig11)')
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figures/fig10_abc_recon.png', bbox_inches='tight', dpi=120)
print('wrote figures/fig10_abc_recon.png')

print('\nkeep-rate of the no-leakage quality filter (filter = the budget-b baseline itself):')
for b in BUDGETS:
    ks = np.mean([r['keep_rate_pool'] for r in R if r['budget'] == b])
    bp, cp = stat('B_pool', b)[0], stat('C_pool', b)[0]
    print(f'  budget {int(b*100):3d}%: keeps {100*ks:5.1f}% of a pool that is 100% correctly '
          f'labelled  -> filtering COSTS {bp-cp:+.3f} macro-F1')
