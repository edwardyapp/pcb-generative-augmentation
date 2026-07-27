# PROVENANCE

Record of what produced the results in two papers, compiled 2026-07-27.

This repository began as a clone of
[rosinality/vq-vae-2-pytorch](https://github.com/rosinality/vq-vae-2-pytorch)
on 2024-11-30. **For both papers' entire working periods it was never committed
to**: the first 18 commits are upstream's, dated 2019-06-10 to 2021-01-23,
ending at `ef5f67c`. Neither paper's development history exists in git.

On 2026-07-27 the tier-1 source state was committed for the first time, as the
single commit tagged `iceta-2026-submission`. Two annotated tags exist:

| tag | commit | what it marks |
|---|---|---|
| `icce-tw-2026` | `ef5f67c` (2021-01-23) | **boundary marker only** — upstream's code at the clone point. Does *not* contain the ICCE-TW working state, because no commit does. See §2.2 for what it does establish. |
| `iceta-2026-submission` | the 2026-07-27 commit | the ICETA submission state: code, `results/`, `figures/` |

Tags are referenced by name throughout this document rather than by commit hash,
so that the references stay valid if the commits are ever recreated.

Everything below is therefore reconstructed from file modification times,
bytecode caches, run logs, and the contents of the result files themselves —
not from commit history. Statements that depend on mtimes are flagged **[mtime]**.
mtimes are mutable filesystem metadata and carry no cryptographic guarantee.

---

## 1. The two states

| | ICCE-TW 2026 | ICETA 2026 |
|---|---|---|
| Working period **[mtime]** | 2024-11-30 → 2024-12-23 | 2026-07-11 → 2026-07-19 |
| Committed while in progress? | No | No |
| End state in git? | Scripts only, at `iceta-2026-submission` | Yes, at `iceta-2026-submission` |
| Crop scale | 600 px | 128 px tight → 256 |
| Boards used | all 10 | 8 train / 2 held out (06, 09) |
| Result files | none survive (see §2.4) | `results/` (62 files) |

The two periods are separated by an ~18-month gap with no file activity, so the
boundary itself is unambiguous even though it is not recorded in git. The only
intervening activity is three demo scripts dated 2026-07-02 (`generate_pool.py`,
`demo_app.py`, `make_timelapse.py`), written for a talk and belonging to neither
paper.

---

## 2. ICCE-TW 2026

### 2.1 Scripts

Unmodified from upstream: `vqvae.py`, `train_pixelsnail.py`, `extract_code.py`,
`scheduler.py`, `distributed/`, `pixelsnail_mnist.py` (all mtime 2024-11-30, the
clone date).

Written or modified for the paper **[mtime]**:

| file | mtime |
|---|---|
| `train_vqvae.py` | 2024-12-01 |
| `train_vqvae-trainVal-allTransformations.py` | 2024-12-01 |
| `sample.py` | 2024-12-04 |
| `augmentMVTecImageFiles.py` | 2024-12-04 |
| `randomlyCopyImageFiles.py` | 2024-12-04 |
| `augmentGoodPCBImages.py` | 2024-12-23 |

### 2.2 `dataset.py` and `pixelsnail.py` — recovered

`dataset.py` and `pixelsnail.py` were both edited in place during the ICETA work
(2026-07-11 and 2026-07-12) and never committed, so their Dec-2024 contents are
not directly retrievable. They are nevertheless **recoverable by inference**, and
the answer is that they were never modified for ICCE-TW at all:

`__pycache__/*.cpython-39.pyc`, written 2024-12-01 during the ICCE-TW run,
carries a header recording each source file's size and mtime at compile time:

| file | size in Dec-2024 `.pyc` | size at `HEAD` (upstream) | size now |
|---|---|---|---|
| `vqvae.py` *(control — git-unmodified)* | 7 799 | 7 799 | 7 799 |
| `scheduler.py` *(control — git-unmodified)* | 9 911 | 9 911 | 9 911 |
| **`pixelsnail.py`** | **12 473** | **12 473** | 14 518 |
| **`dataset.py`** | **1 409** | **1 409** | 1 598 |

All four record source mtime `2024-11-30 16:00:01` — the clone timestamp, the
same mtime still carried by the files that were provably never touched.

**Conclusion: during the ICCE-TW work, `pixelsnail.py` and `dataset.py` were
byte-for-byte upstream's.** The ICCE-TW code state for both is recoverable as
`git show ef5f67c:pixelsnail.py` and `git show ef5f67c:dataset.py`. The ICETA
class-conditioning changes in the current `pixelsnail.py` postdate the ICCE-TW
checkpoints by 18 months, which is consistent with those models being
unconditional (§2.3).

*Strength of this claim.* Source-size agreement plus a clone-era mtime, with two
known-unmodified controls behaving identically, is strong evidence — but it is
not a hash match. Exact bytecode comparison was attempted and is **not
available**: recompiling the known-identical controls under Python 3.9.25 fails
to reproduce their Dec-2024 bytecode, so the original `.pyc` files came from a
different 3.9 micro-version and byte-equality cannot be tested for any file. A
same-length-but-different source is conceivable; given the untouched clone mtime
it is implausible.

PyCharm local history was also checked and holds nothing usable: both
`PyCharmCE2024.2` and `PyCharm2025.1` stores contain path references only, with
no file content (zero occurrences of `import torch`, `def forward`, or `class `
across either `changes.storageData`).

### 2.3 Checkpoints — irreplaceable, archived

Three checkpoints, all verified byte-identical between the working tree and the
off-drive archive on 2026-07-27:

| file | size | MD5 |
|---|---|---|
| `vqvae_560.pt` | 6 107 550 | `ba2932cf30e5096464ead3cd260d4822` |
| `pixelsnail_top_357.pt` | 339 597 968 | `50063dc64d882edb3294887eb2e6f8a5` |
| `pixelsnail_bottom_best.pt` | 359 159 534 | `63fba802a8b1054ec7ac2b09284a43d2` |

Copies exist at `checkpoint/`, `checkpoint (cropped bad PCB)/`, and
`/home/edward-yapp/pcb-checkpoint-archive/`. All three locations agree.

**`/home/edward-yapp/pcb-checkpoint-archive/` is the authoritative copy.** It
holds the three files plus `MANIFEST.md5`, `README.txt`, and `splits.json`; it
lives on `nvme1n1` (`/`), a different physical drive from the project on
`nvme0n1` (`/mnt/storage`). It was created 2026-07-13 after a disk-full event
corrupted checkpoints mid-write. Verify with `md5sum -c MANIFEST.md5` — this
passed on 2026-07-27.

**These are not reproducible.** No training log and no intermediate checkpoints
survive. They were trained on `lmdb/all`, i.e. **all 10 boards including test
boards 06 and 09**, so nothing sampled from them may be used as training data
in any board-split evaluation.

> ⚠️ **Filename collision.** Upstream also ships a file called `vqvae_560.pt` —
> its README describes it as "Checkpoint of VQ-VAE pretrained on FFHQ". That is
> a *different file*: MD5 `11a2bb56500299019ec93a03e5eebbf2`, 6 096 565 bytes.
> The PCB-trained `vqvae_560.pt` above is ours. Any publication of this repo
> must resolve this collision or readers will load the wrong weights.

### 2.4 Results — none survive

No quantitative result file of any kind carries a 2024 mtime. A search of the
whole tree outside the data and virtualenv directories returns only scripts,
checkpoints, IDE config, and `loss_curve.png`. What remains of the ICCE-TW
outputs is image grids only:

- `sample (whole good PCB; epoch: 560; mse: 0.01649; latent: 0.071; avg mse: 0.01649)/`
- `sample (cropped bad PCB; epoch: 560; mse: 0.00165; latent: 0.003; avg mse: 0.00160)/`
- `sample (cropped good PCB, 1024 resolution)/`, `sample (cropped good PCB, 512 resolution)/`
- `sample (whole good PCB; train-validation; all transformations)/`
- `loss_curve.png` (2024-12-01)

The MSE figures quoted in the ICCE-TW paper survive **only as these directory
names**. There is no log or JSON behind them. This is consistent with the paper
having made qualitative claims, but it means **no ICCE-TW number in the paper
can be re-derived from this repository.**

### 2.5 Data

`PCB-cropped/all` — 10 668 files, 1.2 GB, 600×600 crops, filename pattern
`light_NN_<class>_NN_N_600.jpg` across all 10 boards.

**Not reproducible from this repo.** Per `findings.md` §Finding 4, the script
that produced these 600 px crops ran on another machine and is not present here.
The crops are an input, not a derived artefact, as far as this repository is
concerned.

### 2.6 Reproducibility summary — ICCE-TW

| artefact | reproducible here? |
|---|---|
| VQ-VAE / prior model code | **Yes** — `ef5f67c:pixelsnail.py`, `ef5f67c:dataset.py`, `vqvae.py` (§2.2) |
| training / sampling scripts | **Yes** — `train_vqvae.py`, `sample.py` etc. survive in place (§2.1) |
| the three checkpoints | **No** — archived only; no logs, no intermediates |
| 600 px crops | **No** — cropping script ran elsewhere |
| reported MSE numbers | **No** — survive only as directory names |
| sample image grids | Present as files; not regenerable bit-for-bit |

The ICCE-TW **code** is recoverable; its **models and numbers** are not. Retraining
from the recovered code would produce different weights, and there is no logged
metric to check them against. Preservation of the archive in §2.3 is the only
thing standing between the paper and total loss of its models.

---

## 3. ICETA 2026

Claims and numbers are stated in `findings.md`; tables in `results/paper_tables.md`.

### 3.1 Scripts → results → tables

| paper element | produced by | reads / writes |
|---|---|---|
| Tables 1–3, Figs 1–4, 7 | `make_paper_figs.py` | reads `results/classifier_A_b*_s*.json`, `classifier_Atight_b*_s*.json`, `classifier_Atightpb_b100_s*.json`, `results/manifest_tight.csv`; writes `results/paper_tables.md`, `figures/fig1–4,7` |
| Table 4 / Finding 5a | `cond_check.py`, `cond_check_n360_report.py` | writes `results/cond_checks.jsonl`, `cond_check_ep320_n360.jsonl`, `cond_check_ep320_n360_pooled.json` |
| Finding 5b (A/B/C) | `abc_budget.py` | writes `results/abc_budget_b{10,25,50,100}.json`, `abc_budget_b10_seeds.json` |
| Finding 5c (leakage ceiling) | `abc_recon.py` | writes `results/abc_recon.json` |
| Fig 11 (honest vs leaky) | `plot_budget_abc.py` | reads `results/abc_recon.json`, `abc_budget_b*.json`; writes `figures/fig11_budget_abc.png` |
| Finding 6 (binary track) | `abc_binary.py` | writes `results/abc_binary.json` |
| Fig 8, 9, 12 | `plot_loss_curves.py`, `plot_consistency.py`, `plot_nll.py` | write `figures/fig8,9,12` |

Supporting pipeline: `build_split.py` (board split → `results/splits.json`),
`make_tight_crops.py` (tight crops + `results/manifest_tight*.csv`),
`train_classifier.py`, `train_vqvae_tight.py`, `train_pixelsnail_cond.py`,
`extract_code_labeled.py`, `make_budget_subsamples.py`, `train_binary.py`,
`make_defect_detector.py`, `detector_600.py`, `calibrate_sliding.py`,
`nn_baseline.py`, `nn_baseline_600.py`, `nn_pixel.py`, `find_duplicates.py`,
`eval_prior_nll.py`, `gate_check.py`, `pcb_utils.py`. Driver shell scripts:
`run_budget_pipeline.sh`, `run_budget_pipeline_resume.sh`, `run_binary_track.sh`,
`run_b10_seeds.sh`, `run_budget_b25_b50.sh`, `run_cond_check_n360.sh`,
`run_cond_checks.sh`, `run_converge.sh`, `run_nll_eval.sh`, `run_genreview.sh`,
`run_reorder.sh`, `run_samples_review.sh`, `status.sh`.

### 3.2 Data

| dir | files | size | origin |
|---|---|---|---|
| `VOC_PCB/` | 10 668 images + VOC annotations | 1.2 GB | HRIPCB, as distributed (includes the dataset's own `generate_txt.py`, `check_xml.py`) |
| `PCB-cropped/all` | 10 668 | 1.2 GB | 600 px crops — **input, not reproducible here** (§2.5) |
| `PCB-cropped-tight/all` | 5 416 | 427 MB | derived by `make_tight_crops.py` from VOC bboxes — **reproducible** |
| `PCB-nodefect/` | 5 416 | 407 MB | derived by the binary track — **reproducible** |
| `lmdb/` | — | 2.4 GB | extracted codes — **reproducible** via `extract_code_labeled.py` |

Board split (`results/splits.json`, seed 0): test = boards 06, 09; train = 01,
04, 05, 07, 08, 10, 11, 12. 10 668 crops → 8 596 train / 2 072 test. Six
classes: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper.

### 3.3 Checkpoints

236 `.pt` files, 51 GB, in `checkpoint/`. Naming: `prior_b{10,25,50,100}_{top,bottom}_NNN.pt`
(honest per-budget priors), `uncond_*` (unconditional/binary track),
`pixelsnail_cond*`/`condtight*` (class-conditional priors),
`classifier_A*_b*_s*.pt`, `filter_b*_s*.pt`, `defect_detector*.pt`.
`results/checkpoint_inventory.csv` (from `checkpoint_inventory.py`) indexes them.

These are **not archived off-drive** and are **not in git**. They are in
principle reproducible by re-running the pipeline, at substantial GPU cost;
`logs/` retains the training logs (72 MB) that the ICCE-TW run lacks.

### 3.4 Reproducibility summary — ICETA

| artefact | reproducible here? |
|---|---|
| all tables and figures from `results/` | **Yes** — scripts present, mapping in §3.1 |
| `results/` from checkpoints + data | **Yes** |
| checkpoints from data | **Yes in principle** — logs survive; GPU-expensive |
| tight crops, no-defect crops, lmdb, splits | **Yes** |
| 600 px crops (`PCB-cropped/`) | **No** — upstream input, cropping script absent |

The ICETA chain is reproducible end-to-end **given `PCB-cropped/` as a starting
input**. That one link is inherited from the ICCE-TW era and is the only break.

---

## 4. What cannot be reconstructed — plainly stated

1. **The ICCE-TW checkpoints' training provenance.** No logs, no intermediate
   checkpoints, no hyperparameter record beyond what the surviving scripts imply.
2. **The ICCE-TW reported numbers.** Survive only as directory names.
3. **The 600 px cropping procedure.** Ran on another machine; not in this repo.
4. **Which exact script version produced any given 2024 artefact.** With no
   commits in the working period, every mapping in §2 rests on mtimes alone.

Resolved, and no longer on this list: the Dec-2024 state of `pixelsnail.py` and
`dataset.py`, recovered by inference in §2.2 — both were upstream's, unmodified.

Dates in §1 and §2 marked **[mtime]** are filesystem metadata, which is mutable
and carries no cryptographic guarantee. They are consistent across ~60 files and
with a clean 18-month gap, which is good evidence, but it is not the same as a
commit.
