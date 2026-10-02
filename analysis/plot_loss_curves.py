"""Parse the PixelSNAIL training logs into per-epoch loss/acc curves, plot them, and
test the question that actually matters: had the tight prior CONVERGED at epoch 80, or
was its loss still falling — in which case the conditioning check was premature?

The trainers only ever wrote loss to a tqdm progress bar, so the curve has to be
recovered from the tqdm lines in the redirected logs. Each line is the instantaneous
batch loss; we de-duplicate on (hier, epoch, iteration) and take the per-epoch mean.

The ORIGINAL 420-epoch run (Dec 2024) left NO log and only two terminal checkpoints
(top@357, bottom@best), so its curve is genuinely unrecoverable — we say so rather than
invent one, and anchor it instead by evaluating those final weights (see eval_priors.py).
"""
import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LINE = re.compile(r'ep (\d+) (top|bottom) loss ([\d.]+) acc ([\d.]+).*?\|\s*(\d+)/(\d+)')

RUNS = {
    'condtight2 (tight codes, STRONG conditioning, 80ep)': 'logs/rebase_tight2_5090.log',
    'condtight (tight codes, weak conditioning, 60ep)': 'logs/rebase_tight_5090.log',
    'cond600 (600px codes, killed)': 'logs/prior_5090.log',
}
COL = {'top': '#2C7FB8', 'bottom': '#C0473B'}


def parse(path):
    """-> {hier: {epoch: [losses]}}, de-duplicated on (hier, epoch, iter)."""
    if not os.path.exists(path):
        return {}
    seen, out = set(), {}
    with open(path, errors='ignore') as fh:
        for raw in re.split(r'[\r\n]', fh.read()):   # tqdm may use either separator
            m = LINE.search(raw)
            if not m:
                continue
            ep, hier, loss, acc, it = int(m[1]), m[2], float(m[3]), float(m[4]), int(m[5])
            key = (hier, ep, it)
            if key in seen:
                continue
            seen.add(key)
            out.setdefault(hier, {}).setdefault(ep, []).append((loss, acc))
    return out


def curve(d, hier):
    eps = sorted(d.get(hier, {}))
    if not eps:
        return np.array([]), np.array([]), np.array([])
    loss = np.array([np.mean([x[0] for x in d[hier][e]]) for e in eps])
    acc = np.array([np.mean([x[1] for x in d[hier][e]]) for e in eps])
    return np.array(eps), loss, acc


def converged(eps, loss, tail=20):
    """Is it flat at the end? Fit a line to the last `tail` epochs."""
    if len(eps) < tail + 5:
        return None
    e, l = eps[-tail:], loss[-tail:]
    slope = np.polyfit(e, l, 1)[0]                      # loss units per epoch
    drop = l[0] - l[-1]                                 # absolute fall over the tail
    pct = 100 * drop / l[0] if l[0] else 0
    # project: at this slope, how much more would another `tail` epochs buy?
    return {'slope': slope, 'drop': drop, 'pct': pct, 'proj': -slope * tail,
            'first': l[0], 'last': l[-1]}


def main():
    os.makedirs('figures', exist_ok=True)
    data = {name: parse(p) for name, p in RUNS.items()}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    verdicts = {}
    for col, hier in enumerate(['top', 'bottom']):
        axl, axa = axes[0][col], axes[1][col]
        for name, d in data.items():
            eps, loss, acc = curve(d, hier)
            if not len(eps):
                continue
            main_run = name.startswith('condtight2')
            axl.plot(eps, loss, lw=2.2 if main_run else 1.2,
                     alpha=1.0 if main_run else 0.45,
                     color=COL[hier] if main_run else '#888',
                     label=name.split(' (')[0] + f'  (n={len(eps)}ep)')
            axa.plot(eps, acc, lw=2.2 if main_run else 1.2,
                     alpha=1.0 if main_run else 0.45,
                     color=COL[hier] if main_run else '#888',
                     label=name.split(' (')[0])
            if main_run:
                v = converged(eps, loss)
                verdicts[hier] = v
                if v:
                    axl.plot(eps[-20:], np.polyval(np.polyfit(eps[-20:], loss[-20:], 1), eps[-20:]),
                             'k--', lw=1.6, label='fit, last 20 ep')
        axl.set_title(f'{hier.upper()} prior — training loss'); axl.set_ylabel('loss (per-epoch mean)')
        axa.set_title(f'{hier.upper()} prior — code accuracy'); axa.set_ylabel('teacher-forced acc')
        for a in (axl, axa):
            a.set_xlabel('epoch'); a.grid(alpha=0.3); a.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('figures/fig8_prior_loss_curves.png', bbox_inches='tight', dpi=120)
    plt.close()

    print('=' * 78)
    print('CONVERGENCE TEST — condtight2 (the prior the conditioning check was run on)')
    print('=' * 78)
    for hier, v in verdicts.items():
        if not v:
            print(f'{hier}: too few epochs to test'); continue
        still = v['slope'] < -1e-4
        print(f'\n{hier.upper()} prior, last 20 epochs (61->80):')
        print(f'  loss {v["first"]:.4f} -> {v["last"]:.4f}   (fell {v["drop"]:.4f}, {v["pct"]:.1f}%)')
        print(f'  slope           = {v["slope"]:+.5f} loss/epoch')
        print(f'  another 20 ep   ~ {v["proj"]:.4f} further drop at this rate')
        print(f'  VERDICT: {"STILL FALLING — not converged" if still else "flat — converged"}')
    print('\nORIGINAL 420-epoch run: no log survives and only 2 terminal checkpoints exist')
    print('(top@357, bottom@best) -> its loss curve is NOT recoverable. Anchored by direct')
    print('evaluation of the final weights instead (eval_priors.py).')
    print('\nwrote figures/fig8_prior_loss_curves.png')
    np.save('results/prior_curves.npy',
            {n: {h: curve(d, h) for h in ('top', 'bottom')} for n, d in data.items()},
            allow_pickle=True)


if __name__ == '__main__':
    main()
