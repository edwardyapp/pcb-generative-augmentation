"""Train vs held-out NLL across epochs — convergence and memorisation in one figure.

Read it like this:
  * held-out NLL still falling at 320  -> not converged; the epoch-320 gate was still premature
  * held-out NLL flat while train falls -> converged, and the prior is now memorising
  * held-out NLL RISING while train falls -> overfitting outright
The gap (held-out - train) is the memorisation gap, but it is an UPPER bound: the held-out set
is the TEST BOARDS, so part of any gap is board shift rather than memorisation.
"""
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = 'results/prior_nll.json'
if not os.path.exists(SRC):
    raise SystemExit('no results/prior_nll.json yet')

rows = json.load(open(SRC))
eps = [r['epoch'] for r in rows]
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

for ax, hier, col in ((axes[0], 'top', '#2C7FB8'), (axes[1], 'bottom', '#C0473B')):
    tr = [r[hier]['train_nll'] for r in rows]
    te = [r[hier]['heldout_nll'] for r in rows]
    ax.plot(eps, tr, 'o-', lw=2.2, color=col, label='train (IN-SAMPLE)')
    ax.plot(eps, te, 's--', lw=2.2, color=col, alpha=0.55, label='held-out (test boards 06/09)')
    ax.fill_between(eps, tr, te, color=col, alpha=0.12)
    ax.set_title(f'{hier.upper()} prior — NLL (nats/code)')
    ax.set_xlabel('prior training epoch (absolute)')
    ax.set_ylabel('NLL (nats/code)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

ax = axes[2]
for hier, col in (('top', '#2C7FB8'), ('bottom', '#C0473B')):
    ax.plot(eps, [r[hier]['gap'] for r in rows], 'o-', lw=2.2, color=col,
            label=f'{hier}: held-out − train')
ax.axhline(0, ls='--', c='gray', lw=1)
ax.set_title('Memorisation gap\n(upper bound — held-out set is a different BOARD)')
ax.set_xlabel('prior training epoch (absolute)')
ax.set_ylabel('NLL gap (nats/code)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig12_prior_nll.png', bbox_inches='tight', dpi=120)
print('wrote figures/fig12_prior_nll.png')
