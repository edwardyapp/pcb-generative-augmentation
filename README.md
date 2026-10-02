# What Helps PCB Defect Classification: Crop Scale, Not Generative Augmentation

Code and results for two papers on PCB defect generation and classification,
built on [rosinality/vq-vae-2-pytorch](https://github.com/rosinality/vq-vae-2-pytorch):

- **ICCE-TW 2026** — VQ-VAE-2 + PixelSNAIL generation of PCB defect imagery.
- **ICETA 2026** — a leakage-controlled re-evaluation of that pipeline. Headline:
  a size-matched change of crop scale is worth **+0.65 macro-F1**, while the
  class-conditional generator yields **no usable labeled synthetic defects**.

Findings and the full argument: [`findings.md`](findings.md).
Tables: [`results/paper_tables.md`](results/paper_tables.md).
What produced what, and what cannot be reproduced: **[`PROVENANCE.md`](PROVENANCE.md)** — read this first.

> ### ⚠️ `vqvae_560.pt` is not the upstream FFHQ checkpoint
>
> Upstream shipped a file named `vqvae_560.pt` containing a VQ-VAE pretrained on
> FFHQ. **That file has been removed from this repository** to prevent a name
> collision. Our `vqvae_560.pt` is a *different* model — trained on PCB crops,
> MD5 `ba2932cf30e5096464ead3cd260d4822`, 6 107 550 bytes — and is not in git.
> No public checkpoint release exists.
> Upstream's FFHQ file (MD5 `11a2bb56500299019ec93a03e5eebbf2`) remains
> available in this repository's history at tag `icce-tw-2026`, and from upstream.

## What is in this repository

Code, result files and figures only — about 10 MB.

| path | contents |
|---|---|
| `*.py`, `run_*.sh` | pipeline, training, evaluation and plotting scripts |
| `results/` | every number in both papers, as JSON/CSV/JSONL |
| `figures/` | figures 1–13 |
| `findings.md` | the claims, with the evidence for each |
| `PROVENANCE.md` | provenance, reproducibility, and what is lost |

**Not in git** (`PROVENANCE.md` describes each): model weights (51 GB), the
HRIPCB-derived image data, extracted `lmdb` codes, training logs. None of these
is publicly released.

## Data

Derived from the public **HRIPCB** dataset (10 boards, 6 defect classes:
missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper).

Board-level split, seed 0 — boards **06 and 09 held out** as test, touched by
nothing: not the classifier, not the VQ-VAE, not the priors. 10 668 crops →
8 596 train / 2 072 test. Recorded in `results/splits.json`.

## Reproducing the tables

Given the crops and checkpoints in place:

```
python build_split.py          # -> results/splits.json
python make_tight_crops.py     # -> PCB-cropped-tight/, results/manifest_tight*.csv
python make_paper_figs.py      # -> results/paper_tables.md, figures/fig1-4,7
python plot_budget_abc.py      # -> figures/fig11_budget_abc.png
python cond_check_n360_report.py
```

Full script-to-table mapping: `PROVENANCE.md` §3.1.

One link in the chain is **not** reproducible here: the 600 px crops in
`PCB-cropped/` were produced by a script that ran on another machine and is not
in this repository. They are an input and are not in this repository. See
`PROVENANCE.md` §2.5.

## Upstream usage

The VQ-VAE-2 / PixelSNAIL implementation is upstream's, with modifications for
class-conditional priors. Upstream's original instructions:

1. Stage 1 (VQ-VAE): `python train_vqvae.py [DATASET PATH]`
2. Extract codes: `python extract_code.py --ckpt checkpoint/[VQ-VAE CHECKPOINT] --name [LMDB NAME] [DATASET PATH]`
3. Stage 2 (PixelSNAIL): `python train_pixelsnail.py [LMDB NAME]`

Requires Python ≥ 3.6, PyTorch ≥ 1.1, lmdb. Developed against Python 3.9.

## Citing

If you use this code, please cite the relevant paper (titles in `PROVENANCE.md`
§0).

## License

MIT. The VQ-VAE-2 / PixelSNAIL implementation is Copyright (c) 2019 Kim
Seonghyeon; modifications and all PCB-specific work are Copyright (c) 2026
Edward Yapp. See [`LICENSE`](LICENSE).

The HRIPCB dataset is the property of its original authors and is subject to its
own terms. This repository does not redistribute the HRIPCB images or the crops
derived from them; it contains their file names, splits and labels
(`results/*.csv`, `results/splits.json`), and a small number of example crops
appear in the figures (`figures/`, `fig_samples.*`).

Development of this code was assisted by AI coding tools.
