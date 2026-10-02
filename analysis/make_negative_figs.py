"""Figures for the generative negative result:
  fig5 - generated samples, one row per conditioned class (visually indistinguishable)
  fig6 - conditioning confusion heatmap (collapse to a single attractor)
"""
import sys, pathlib  # repo layout (README.md): src/, experiments/, analysis/ importable from anywhere
sys.path[1:1] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('src', 'experiments', 'analysis')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from pcb_utils import CLASSES

plt.rcParams.update({'font.size': 10, 'figure.dpi': 120})

# --- fig5: top row (8 samples) of each class grid, stacked ---
fig, axes = plt.subplots(6, 1, figsize=(12, 9.6))
for i, c in enumerate(CLASSES):
    g = Image.open(f'synth_check2/grid_{c}.png').convert('RGB')
    strip = g.crop((0, 0, g.width, 260))  # first row of the nrow=8 grid
    axes[i].imshow(strip)
    axes[i].set_xticks([]); axes[i].set_yticks([])
    axes[i].set_ylabel(c.replace('_', '\n'), rotation=0, ha='right', va='center',
                       fontsize=9, fontweight='bold')
fig.suptitle('Generated crops conditioned on each defect class (stronger conditioning, temp 1.0)\n'
             'Samples are generic PCB texture — the class-specific defect is not rendered',
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('figures/fig5_generated_samples.png', bbox_inches='tight')
plt.close()

# --- fig6: conditioning confusion heatmap ---
conf = np.array([
    [6, 8, 1, 1, 14, 2],
    [1, 8, 0, 0, 22, 1],
    [0, 8, 0, 0, 21, 3],
    [1, 8, 0, 0, 23, 0],
    [0, 6, 1, 0, 22, 3],
    [0, 9, 0, 0, 16, 7],
], float)
row = conf / conf.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(7, 5.8))
im = ax.imshow(row, cmap='magma', vmin=0, vmax=1)
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels([c.replace('_', '\n') for c in CLASSES], fontsize=8)
ax.set_yticklabels(CLASSES, fontsize=8)
ax.set_xlabel('predicted class (ResNet-18 trained on real crops)')
ax.set_ylabel('conditioned class')
for i in range(6):
    for j in range(6):
        ax.text(j, i, f'{row[i,j]:.2f}', ha='center', va='center', fontsize=8,
                color='white' if row[i, j] < 0.5 else 'black')
acc = np.trace(conf) / conf.sum()
ax.set_title('Conditioning failure: every class collapses to the same output\n'
             f'consistency {100*acc:.1f}% vs {100/6:.1f}% chance '
             '(a diagonal would mean conditioning worked)', fontsize=11)
plt.colorbar(im, label='fraction of samples')
plt.tight_layout()
plt.savefig('figures/fig6_conditioning_confusion.png', bbox_inches='tight')
plt.close()
print('wrote figures/fig5_generated_samples.png, figures/fig6_conditioning_confusion.png')
print(f'consistency {100*acc:.1f}% vs chance {100/6:.1f}%')
