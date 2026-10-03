# What Helps PCB Defect Classification: Crop Scale, Not Generative Augmentation

## Repository layout

| path | what lives there |
|---|---|
| `src/` | models (VQ-VAE-2, PixelSNAIL), data preparation, training, shared utilities |
| `experiments/` | the ICETA experiments (budget A/B/C, conditioning checks, sampling, detectors) and their `run_*.sh` drivers |
| `analysis/` | figures, tables, review pages, status reports |
| `legacy/` | ICCE-TW-era scripts and the talk demo ([`legacy/DEMO_README.md`](legacy/DEMO_README.md)) |
| `results/` | every number in both papers, as JSON/CSV/JSONL |
| `figures/`, `fig_samples.*` | figures 1–13; the paper's Fig. 1 |
| `paper/` | the ICETA 2026 camera-ready ([`paper/ICETA-2026.pdf`](paper/ICETA-2026.pdf)) |
| [`findings.md`](findings.md) · [`PROVENANCE.md`](PROVENANCE.md) · [`REPRODUCE.md`](REPRODUCE.md) · [`AUDIT.md`](AUDIT.md) | the claims · what produced what · how to rerun each claim · independent verification |

Run every script from the repository root (e.g. `python experiments/abc_budget.py`);
data paths are relative to it. Scripts import each other across these directories
by module name, so no install step is needed.

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

Code, result files, figures and the ICETA paper — about 11 MB; see [Repository layout](#repository-layout).

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
python src/build_split.py                     # -> results/splits.json
python src/make_tight_crops.py                # -> PCB-cropped-tight/, results/manifest_tight*.csv
python analysis/make_paper_figs.py            # -> results/paper_tables.md, figures/fig1-4,7
python analysis/plot_budget_abc.py            # -> figures/fig11_budget_abc.png
python experiments/cond_check_n360_report.py
```

Full script-to-table mapping: `PROVENANCE.md` §3.1.

The chain starts from public data. The 600 px crops in `PCB-cropped/all` are a
byte-identical copy of `JPEGImages/` in the TDD-net VOC release (`VOC_PCB`),
itself derived from HRIPCB: all 10 668 files match by name and MD5. Download that
release and use its `JPEGImages/` as `PCB-cropped/all`. (Corrected 2026-10-02: this
README previously said the crops came from a lost script; see `PROVENANCE.md` §5.4.)

## Upstream usage

The VQ-VAE-2 / PixelSNAIL implementation is upstream's, with modifications for
class-conditional priors. Upstream's original instructions:

1. Stage 1 (VQ-VAE): `python src/train_vqvae.py [DATASET PATH]`
2. Extract codes: `python src/extract_code.py --ckpt checkpoint/[VQ-VAE CHECKPOINT] --name [LMDB NAME] [DATASET PATH]`
3. Stage 2 (PixelSNAIL): `python src/train_pixelsnail.py [LMDB NAME]`

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
