# Crop-scale paper — tables

## Table 1: Crop scale @ 100% budget (macro-F1, 3 seeds, board-split)

| set | n_train | macro-F1 | accuracy |
|---|---|---|---|
| 600px baseline | 2149 | 0.246 ± 0.013 | 0.315 ± 0.004 |
| TIGHT matched | 2149 | 0.897 ± 0.020 | 0.896 ± 0.021 |
| TIGHT per-bbox | 4274 | 0.947 ± 0.005 | 0.947 ± 0.005 |

## Table 2: Budget curves (macro-F1)

| budget | 600px | TIGHT |
|---|---|---|
| 10% | 0.144 ± 0.035 | 0.422 ± 0.036 |
| 25% | 0.191 ± 0.029 | 0.640 ± 0.010 |
| 50% | 0.252 ± 0.016 | 0.779 ± 0.007 |
| 100% | 0.246 ± 0.013 | 0.897 ± 0.020 |

## Table 3: Per-class F1 @ 100% (600px vs tight matched)

| class | 600px | tight |
|---|---|---|
| missing_hole | 0.777 | 1.000 |
| mouse_bite | 0.247 | 0.844 |
| open_circuit | 0.135 | 0.891 |
| short | 0.188 | 0.935 |
| spur | 0.000 | 0.908 |
| spurious_copper | 0.131 | 0.807 |

## Table 4: Class-conditional generation — conditioning check (NEGATIVE)

Priors: tight codes, class injected 3 ways (feature bias + per-PixelBlock
condition + output-logit bias), 80 epochs. 32 samples/class, temp 1.0,
decoded with the tight VQ-VAE and classified by the tight ResNet-18
(which scores 0.918 macro-F1 on *reconstructed real* crops — so the
VQ-VAE and the classifier are both sound).

| metric | value | chance |
|---|---|---|
| conditioning consistency | 22.4% (43/192) | 16.7% |
| consistency excluding the collapse class (spur) | **13.1%** (21/160) | 16.7% |
| samples predicted "spur" regardless of conditioning | **61.5%** (118/192) | 16.7% |

The 22.4% headline is an artifact of mode collapse: the classifier assigns
61.5% of *all* samples to `spur` whatever class was requested, so the `spur`
row alone carries the apparent signal. Removing it, consistency falls *below*
chance. Samples are photorealistic PCB substrate with **no defect rendered**
(Fig 5, Fig 7): `missing_hole` samples show pads with their holes intact.
Replicated in a second independent sampling run (44/192 = 22.9%, same collapse).

| conditioned | predicted spur | predicted correctly |
|---|---|---|
| missing_hole | 14/32 | 6/32 |
| mouse_bite | 22/32 | 8/32 |
| open_circuit | 21/32 | 0/32 |
| short | 23/32 | 0/32 |
| spur | 22/32 | 22/32 |
| spurious_copper | 16/32 | 7/32 |
