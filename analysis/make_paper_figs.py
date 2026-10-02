"""Assemble all figures + tables for the crop-scale paper (stands alone regardless of
whether the generator fix lands). Reads the classifier result JSONs + VOC annotations.
Outputs to figures/ and results/paper_tables.md.
"""
import sys, pathlib  # repo layout (README.md): src/, experiments/, analysis/ importable from anywhere
sys.path[1:1] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('src', 'experiments', 'analysis')]
import json
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pcb_utils import CLASSES, parse_filename

C600 = '#C0473B'   # 600px
CTIGHT = '#2C7FB8'  # tight matched
CPB = '#3B9C5A'     # tight per-bbox
plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3,
                     'axes.axisbelow': True, 'figure.dpi': 120})


def cell(patt):
    ds = [json.load(open(f)) for f in glob.glob(patt)]
    return ds or None


def mstd(ds, key='macro_f1'):
    v = [d[key] for d in ds]
    return np.mean(v), np.std(v)


def perclass(ds):
    return {c: np.mean([d['per_class_f1'][c] for d in ds]) for c in CLASSES}


BUDGETS = [('10', 0.1), ('25', 0.25), ('50', 0.5), ('100', 1.0)]


def fig1_cropscale():
    a = cell('results/classifier_A_b100_s*.json')
    t = cell('results/classifier_Atight_b100_s*.json')
    p = cell('results/classifier_Atightpb_b100_s*.json')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    # left: overall macro-F1
    names = ['600px\n(baseline)', 'TIGHT\nmatched', 'TIGHT\nper-bbox']
    means = [mstd(a)[0], mstd(t)[0], mstd(p)[0]]
    stds = [mstd(a)[1], mstd(t)[1], mstd(p)[1]]
    ax[0].bar(names, means, yerr=stds, capsize=6, color=[C600, CTIGHT, CPB])
    ax[0].axhline(1/6, ls='--', c='gray', lw=1)
    ax[0].text(2.4, 1/6+0.01, 'chance', color='gray', ha='right', fontsize=9)
    for i, m in enumerate(means):
        ax[0].text(i, m+0.02, f'{m:.2f}', ha='center', fontweight='bold')
    ax[0].set_ylabel('macro-F1 (test boards 06+09)')
    ax[0].set_title('(a) Crop scale, 100% budget (n_train=2149 for both 600px & matched)')
    ax[0].set_ylim(0, 1.05)
    # right: per-class F1, 600px vs tight matched
    pa, pt = perclass(a), perclass(t)
    x = np.arange(6)
    ax[1].bar(x-0.2, [pa[c] for c in CLASSES], 0.4, label='600px', color=C600)
    ax[1].bar(x+0.2, [pt[c] for c in CLASSES], 0.4, label='TIGHT matched', color=CTIGHT)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([c.replace('_', '\n') for c in CLASSES], fontsize=8)
    ax[1].set_ylabel('per-class F1')
    ax[1].set_title('(b) Per-class repair (same data, only crop scale differs)')
    ax[1].legend()
    ax[1].set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('figures/fig1_cropscale.png', bbox_inches='tight')
    plt.close()


def fig2_budget():
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [b for _, b in BUDGETS]
    for patt, name, col in [('results/classifier_A_b%s_s*.json', '600px baseline', C600),
                            ('results/classifier_Atight_b%s_s*.json', 'TIGHT (corrected)', CTIGHT)]:
        m, s = [], []
        for name_b, _ in BUDGETS:
            ds = cell(patt % name_b)
            mm, ss = mstd(ds)
            m.append(mm); s.append(ss)
        ax.errorbar([b*100 for b in xs], m, yerr=s, marker='o', capsize=4, label=name, color=col, lw=2)
    ax.axhline(1/6, ls='--', c='gray', lw=1)
    ax.text(100, 1/6+0.01, 'chance', color='gray', ha='right', fontsize=9)
    ax.set_xlabel('real-data budget (% of train pool)')
    ax.set_ylabel('macro-F1 (test boards 06+09)')
    ax.set_title('Macro-F1 vs data budget: 600px vs tight crops (3 seeds)')
    ax.set_xticks([10, 25, 50, 100])
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig('figures/fig2_budget_curves.png', bbox_inches='tight')
    plt.close()


def fig3_confound():
    a = mstd(cell('results/classifier_A_b100_s*.json'))[0]
    t = mstd(cell('results/classifier_Atight_b100_s*.json'))[0]
    p = mstd(cell('results/classifier_Atightpb_b100_s*.json'))[0]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    stages = ['600px\nbaseline', '+ crop scale\n(matched, same n)', '+ dataset size\n(per-bbox, 2x n)']
    vals = [a, t, p]
    bars = ax.bar(stages, vals, color=[C600, CTIGHT, CPB])
    ax.annotate('', xy=(1, t), xytext=(1, a), arrowprops=dict(arrowstyle='->', color='k'))
    ax.text(1.05, (a+t)/2, f'+{t-a:.2f}\n(scale, +{(t-a)/(t-a+p-t)*100:.0f}% of gain)', va='center', fontsize=9)
    ax.text(2.05, (t+p)/2, f'+{p-t:.2f}\n(size, +{(p-t)/(t-a+p-t)*100:.0f}% of gain)', va='center', fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v+0.02, f'{v:.2f}', ha='center', fontweight='bold')
    ax.set_ylabel('macro-F1')
    ax.set_title('Effect decomposition: crop scale dominates dataset size')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('figures/fig3_confound.png', bbox_inches='tight')
    plt.close()


def fig4_crop_policy():
    # defect distance-from-centre over all plain crops
    near = []
    allc = []
    for x in glob.glob('VOC_PCB/Annotations/*.xml'):
        p = parse_filename(os.path.basename(x).replace('.xml', '.jpg'))
        if p is None or p['variant'] != 'plain':
            continue
        root = ET.parse(x).getroot()
        cens = []
        for o in root.findall('object'):
            b = o.find('bndbox')
            cx = (int(b.find('xmin').text)+int(b.find('xmax').text))/2
            cy = (int(b.find('ymin').text)+int(b.find('ymax').text))/2
            cens.append((cx, cy)); allc.append(np.hypot(cx-300, cy-300))
        near.append(min(np.hypot(cx-300, cy-300) for cx, cy in cens))
    near = np.array(near)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(near, bins=40, color=CTIGHT, alpha=0.85, label='nearest defect per crop')
    ax.axvline(np.median(near), c='k', lw=2, label=f'median {np.median(near):.0f}px')
    ax.axvline(230, c='red', ls='--', lw=2, label='uniform-random null (~230px)')
    ax.axvline(50, c='green', ls=':', lw=2, label='"centred" would be <50px')
    ax.set_xlabel('defect distance from crop centre (px, 600x600 crop)')
    ax.set_ylabel('# crops')
    ax.set_title('Crop policy recovery: defects are NOT centred\n'
                 f'(only {100*(near<50).mean():.0f}% within 50px; median {np.median(near):.0f}px ~ random)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/fig4_crop_policy.png', bbox_inches='tight')
    plt.close()
    return near


def tables():
    lines = ['# Crop-scale paper — tables\n']
    lines.append('## Table 1: Crop scale @ 100% budget (macro-F1, 3 seeds, board-split)\n')
    lines.append('| set | n_train | macro-F1 | accuracy |')
    lines.append('|---|---|---|---|')
    for patt, name in [('results/classifier_A_b100_s*.json', '600px baseline'),
                       ('results/classifier_Atight_b100_s*.json', 'TIGHT matched'),
                       ('results/classifier_Atightpb_b100_s*.json', 'TIGHT per-bbox')]:
        ds = cell(patt); m, s = mstd(ds); ma, sa = mstd(ds, 'accuracy')
        lines.append(f"| {name} | {ds[0]['n_train']} | {m:.3f} ± {s:.3f} | {ma:.3f} ± {sa:.3f} |")
    lines.append('\n## Table 2: Budget curves (macro-F1)\n')
    lines.append('| budget | 600px | TIGHT |')
    lines.append('|---|---|---|')
    for nb, _ in BUDGETS:
        a = mstd(cell(f'results/classifier_A_b{nb}_s*.json'))
        t = mstd(cell(f'results/classifier_Atight_b{nb}_s*.json'))
        lines.append(f"| {nb}% | {a[0]:.3f} ± {a[1]:.3f} | {t[0]:.3f} ± {t[1]:.3f} |")
    lines.append('\n## Table 3: Per-class F1 @ 100% (600px vs tight matched)\n')
    lines.append('| class | 600px | tight |')
    lines.append('|---|---|---|')
    pa = perclass(cell('results/classifier_A_b100_s*.json'))
    pt = perclass(cell('results/classifier_Atight_b100_s*.json'))
    for c in CLASSES:
        lines.append(f"| {c} | {pa[c]:.3f} | {pt[c]:.3f} |")
    open('results/paper_tables.md', 'w').write('\n'.join(lines) + '\n')


# --- generative arm (negative result) -----------------------------------------
# Confusion of the final stronger-conditioned priors (condtight2, 80ep, temp 1.0),
# 32 samples/class decoded and classified by the tight ResNet-18.
# This is the run recorded in logs/cond_check2.log (43/192 = 22.4%), the one findings.md
# cites. A second, independent sampling run gave 44/192 = 22.9% with the same collapse
# (61.5% of samples -> spur in BOTH runs) -- the result replicates.
COND_CONF = np.array([[6, 8, 1, 1, 14, 2],
                      [1, 8, 0, 0, 22, 1],
                      [0, 8, 0, 0, 21, 3],
                      [1, 8, 0, 0, 23, 0],
                      [0, 6, 1, 0, 22, 3],
                      [0, 9, 0, 0, 16, 7]])


def fig5_generative():
    """Left: real tight crops vs class-conditioned samples. Right: the confusion
    that shows conditioning had no effect (every class collapses onto spur)."""
    import pandas as pd
    from PIL import Image
    df = pd.read_csv('results/manifest_tight.csv')
    fig = plt.figure(figsize=(14, 7.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.18)
    gi = gs[0].subgridspec(6, 7, wspace=0.06, hspace=0.06)
    for r, c in enumerate(CLASSES):
        real = df[df.class_name == c].crop_path.tolist()[:3]
        for j, p in enumerate(real):
            ax = fig.add_subplot(gi[r, j])
            ax.imshow(Image.open(p).resize((128, 128)))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if j == 0:
                ax.set_ylabel(c.replace('_', '\n'), fontsize=7, rotation=0,
                              ha='right', va='center', labelpad=22)
            if r == 0 and j == 1:
                ax.set_title('REAL', fontsize=10, fontweight='bold', color=CTIGHT)
        # 3 tiles sliced out of the 8x4 sample grid (256px tiles, 2px pad)
        g = Image.open(f'synth_check2/grid_{c}.png')
        for j in range(3):
            ax = fig.add_subplot(gi[r, 4 + j])
            t = g.crop((2 + j * 258, 2, 2 + j * 258 + 256, 258)).resize((128, 128))
            ax.imshow(t)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if r == 0 and j == 1:
                ax.set_title('GENERATED (conditioned on this class)',
                             fontsize=10, fontweight='bold', color=C600)
    axc = fig.add_subplot(gs[1])
    P = COND_CONF / COND_CONF.sum(1, keepdims=True)
    im = axc.imshow(P, cmap='Reds', vmin=0, vmax=0.75)
    axc.set_xticks(range(6)); axc.set_yticks(range(6))
    axc.set_xticklabels([c[:9] for c in CLASSES], rotation=45, ha='right', fontsize=8)
    axc.set_yticklabels([c[:12] for c in CLASSES], fontsize=8)
    for i in range(6):
        for j in range(6):
            axc.text(j, i, f'{P[i, j]:.2f}', ha='center', va='center', fontsize=8,
                     color='white' if P[i, j] > 0.4 else 'black')
    axc.set_xlabel('predicted class (tight ResNet-18)')
    axc.set_ylabel('conditioned class')
    d = np.trace(COND_CONF) / COND_CONF.sum()
    axc.set_title(f'Conditioning has no effect\nconsistency {100*d:.1f}% vs 16.7% chance;\n'
                  'every row collapses onto "spur"', fontsize=10)
    axc.grid(False)
    plt.colorbar(im, ax=axc, fraction=0.046, label='P(predicted | conditioned)')
    plt.savefig('figures/fig7_real_vs_generated.png', bbox_inches='tight', dpi=120)
    plt.close()


def table4_conditioning():
    M = COND_CONF
    n, d = M.sum(), np.trace(M)
    si = CLASSES.index('spur')
    keep = [i for i in range(6) if i != si]
    d_ex = sum(M[i, i] for i in keep)
    n_ex = M[keep].sum()
    lines = ['\n## Table 4: Class-conditional generation — conditioning check (NEGATIVE)\n',
             'Priors: tight codes, class injected 3 ways (feature bias + per-PixelBlock',
             'condition + output-logit bias), 80 epochs. 32 samples/class, temp 1.0,',
             'decoded with the tight VQ-VAE and classified by the tight ResNet-18',
             '(which scores 0.918 macro-F1 on *reconstructed real* crops — so the',
             'VQ-VAE and the classifier are both sound).\n',
             '| metric | value | chance |', '|---|---|---|',
             f'| conditioning consistency | {100*d/n:.1f}% ({d}/{n}) | 16.7% |',
             f'| consistency excluding the collapse class (spur) | **{100*d_ex/n_ex:.1f}%** '
             f'({d_ex}/{n_ex}) | 16.7% |',
             f'| samples predicted "spur" regardless of conditioning | **{100*M.sum(0)[si]/n:.1f}%** '
             f'({M.sum(0)[si]}/{n}) | 16.7% |', '',
             f'The {100*d/n:.1f}% headline is an artifact of mode collapse: the classifier assigns',
             '61.5% of *all* samples to `spur` whatever class was requested, so the `spur`',
             'row alone carries the apparent signal. Removing it, consistency falls *below*',
             'chance. Samples are photorealistic PCB substrate with **no defect rendered**',
             '(Fig 5, Fig 7): `missing_hole` samples show pads with their holes intact.',
             'Replicated in a second independent sampling run (44/192 = 22.9%, same collapse).\n',
             '| conditioned | predicted spur | predicted correctly |', '|---|---|---|']
    for i, c in enumerate(CLASSES):
        lines.append(f'| {c} | {M[i, si]}/32 | {M[i, i]}/32 |')
    with open('results/paper_tables.md', 'a') as f:
        f.write('\n'.join(lines) + '\n')
    return 100 * d / n, 100 * d_ex / n_ex


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    fig1_cropscale()
    fig2_budget()
    fig3_confound()
    near = fig4_crop_policy()
    tables()
    fig5_generative()
    cons, cons_ex = table4_conditioning()
    print('figures: fig1_cropscale, fig2_budget_curves, fig3_confound, fig4_crop_policy, '
          'fig7_real_vs_generated  (fig5/fig6 come from make_negative_figs.py)')
    print(f'crop-policy: median nearest-defect {np.median(near):.0f}px, {100*(near<50).mean():.1f}% within 50px')
    print(f'conditioning: {cons:.1f}% consistency ({cons_ex:.1f}% excluding collapse class) vs 16.7% chance')
    print('tables: results/paper_tables.md')
