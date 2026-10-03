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

## 0. Paper titles — authoritative

The submitted manuscripts are `.docx` files and are **not in this repository**.
Their titles are recorded here so the repo has one authoritative source; every
other mention (README H1, tag annotations) is derived from this section.

| paper | title |
|---|---|
| ICETA 2026 | **What Helps PCB Defect Classification: Crop Scale, Not Generative Augmentation** |
| ICCE-TW 2026 | *not recorded here — see §2* |

`findings.md` opens with a **headline**, not a title: "Crop methodology, not
generative AI, is what moves PCB defect classification." It paraphrases the
argument and should not be cited as the paper's name.

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

> **Amended 2026-07-28.** `augmentMVTecImageFiles.py` (this table) and
> `pixelsnail_mnist.py` (the upstream list above) were **removed from git tracking**
> on 2026-07-28 and are no longer in the repository, though both remain on disk.
> The entries above are retained unchanged because they are statements about the
> Dec-2024 working state, which the untracking does not alter. See §5.

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

#### 2.2.1 Where `pixelsnail_bottom_best.pt` came from

`_best` is not a name upstream can produce. Upstream's `train_pixelsnail.py`
saves only `f'checkpoint/pixelsnail_{args.hier}_{str(i+1).zfill(3)}.pt'`, so a
best-checkpoint mechanism appeared to be implied somewhere. It was not. The
suffix is a **manual rename**.

**The `.pyc` method does not apply to the training scripts.** CPython writes
bytecode caches only for *imported* modules, never for the `__main__` entry
point. `__pycache__/` contains `dataset`, `pixelsnail`, `vqvae` and `scheduler`
but no `train_pixelsnail` or `train_vqvae`, and never could have. Their Dec-2024
state had to be established another way.

**`train_pixelsnail.py` was never modified.** Its mtime is
`2024-11-30 16:00:01.729470388` — identical *to the nanosecond* to `vqvae.py`,
and within 3 ms of `scheduler.py`, `extract_code.py` and `pixelsnail_mnist.py`.
That is the signature of a single git checkout writing all of them at once. Its
size, 4 311 bytes, equals `ef5f67c`'s. It carries no best-tracking logic, and
never has.

**`train_vqvae.py` was modified**, on 2024-12-01 15:10, 4 300 → 4 306 bytes. The
whole of that change is two lines, and neither concerns checkpointing:
`range=(-1,1)` → `value_range=(-1,1)` (a torchvision API rename, +6 bytes) and
batch size `128` → `256`.

**What the checkpoint says about itself.** `pixelsnail_bottom_best.pt`
deserialises to `{'model', 'args'}` — upstream's exact save structure — and its
embedded `args` namespace holds exactly upstream's 14 argparse parameters, no
more:

```
batch=32, epoch=420, hier='bottom', lr=0.0003, channel=256, n_res_block=4,
n_res_channel=256, n_out_res_block=0, n_cond_res_block=3, dropout=0.1,
amp='O0', sched=None, ckpt=None, path='lmdb/all'
```

Any code with a best-tracking feature would carry at least one extra flag. There
is none. `pixelsnail_top_357.pt` is identical but for `hier='top'`.

**The `-ad` fork is ruled out.** `../vq-vae-2-pytorch-ad/train_pixelsnail.py`
does implement best-checkpoint saving, but names its output
`f'checkpoint/{object_name}_pixelsnail_{args.hier}_best.pt'` with
`object_name = args.path.split('/')[-1]`, which is never empty for a real path.
It cannot produce a prefix-less `pixelsnail_bottom_best.pt`. Its own
`checkpoint/` holds `pixelsnail_top_420.pt` and `pixelsnail_bottom_420.pt` —
upstream naming, unprefixed, no `_best`. The `vqvae-2` folder
(`github.com/vvvm23/vqvae-2`, active 2024-12-05 → 12-19) is a different
codebase entirely and contains no best-checkpoint logic. No 2024-era shell
script survives in this repository.

**Conclusion.** `pixelsnail_bottom_best.pt` was written by upstream-identical
`train_pixelsnail.py` as `pixelsnail_bottom_NNN.pt` and renamed by hand
afterwards. This matches an established habit visible in
`checkpoint (cropped good PCB 512)/`, which holds
`pixelsnail_bottom_243_lr1e-4.pt` and `pixelsnail_bottom_263_lr2e-4.pt` —
suffixes upstream's f-string cannot generate.

**Still unknown: which epoch.** `args.epoch=420` is the *target* epoch count
passed on the command line, not the epoch at which the file was written; the
loop index is not saved. No training log survives. So "best" is a retrospective
label applied by hand, **not a metric-tracked selection** — nothing records what
it was best *at*, or against what alternatives. The scare quotes around "best"
in the archive's own `README.txt` are consistent with this. Any paper text
implying automatic best-checkpoint selection for this model would be wrong.

*Incidental corroboration.* `path='lmdb/all'` and `ckpt=None` confirm, from
inside the checkpoint, that it was trained on all 10 boards in a single
un-resumed run — a claim that until now rested only on the archive `README.txt`.

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

1. **The ICCE-TW checkpoints' training run.** No logs and no intermediate
   checkpoints survive. The hyperparameters are recoverable — they are embedded
   in the checkpoints themselves (§2.2.1) — but the loss trajectory is not.
2. **The epoch at which `pixelsnail_bottom_best.pt` was saved.** Not recorded in
   the file, and no log survives. "best" is a hand-applied label, not a
   metric-tracked selection (§2.2.1).
3. **The ICCE-TW reported numbers.** Survive only as directory names.
4. **The 600 px cropping procedure.** Ran on another machine; not in this repo.
5. **Which exact script version produced any given 2024 artefact.** With no
   commits in the working period, every mapping in §2 rests on filesystem
   metadata. Established individually for `pixelsnail.py`, `dataset.py`,
   `train_pixelsnail.py` and `train_vqvae.py` (§2.2, §2.2.1); not for the rest.

Resolved, and no longer on this list:

- the Dec-2024 state of `pixelsnail.py` and `dataset.py` — both upstream's,
  unmodified (§2.2);
- the Dec-2024 state of `train_pixelsnail.py` (untouched since checkout) and
  `train_vqvae.py` (two lines, neither checkpoint-related) (§2.2.1);
- the origin of the `_best` filename — a manual rename, not a code feature
  (§2.2.1);
- that the ICCE-TW priors were trained on all 10 boards in one un-resumed run,
  now confirmed from inside the checkpoints rather than from the archive README.

Dates in §1 and §2 marked **[mtime]** are filesystem metadata, which is mutable
and carries no cryptographic guarantee. They are consistent across ~60 files and
with a clean 18-month gap, which is good evidence, but it is not the same as a
commit.

---

## 5. Amendments

This document is amended by appending, never by editing an earlier statement.
Sections 1–4 record what was true when compiled on 2026-07-27 and are left intact
even where a later change has overtaken them; each amendment below says which
statements it overtakes.

### 5.1 Pre-publication cleanup — 2026-07-28

Two commits after the original, both tagged in turn as `iceta-2026-submission`.
No history was rewritten: the 2026-07-27 commit is unchanged and still in the
history, and `icce-tw-2026` still points at `ef5f67c`, byte-exact.

**Files removed from git tracking.** All three belong to neither paper, and no
tracked file imports or invokes any of them. **All three remain on disk** and are
now listed in `.gitignore` so that `git status` stays clean:

| file | referenced in | why untracked |
|---|---|---|
| `augmentMVTecImageFiles.py` | §2.1 table | MVTec `transistor/train/good` → `PCB-cropped-combined`; belongs to neither paper's results |
| `pixelsnail_mnist.py` | §2.1 list, §2.2 | upstream MNIST demo, never used here |
| `eval_demo_pool.py` | — | 2026-07 talk-demo helper, referenced by nothing at all |

Their appearances in §2.1 and §2.2 are **retained deliberately**. Those are
statements about the Dec-2024 working state and about 2024-11-30 mtimes; both
remain true, and the files remain on disk to be inspected. Specifically, the
mtime argument in §2.2 — that `train_pixelsnail.py` shares a nanosecond-identical
mtime with `vqvae.py` and lands within 3 ms of `scheduler.py`, `extract_code.py`
and `pixelsnail_mnist.py`, the signature of a single checkout — is unaffected.

**One removal was reverted.** `randomlyCopyImageFiles.py` (§2.1 table) was
untracked in the first cleanup commit as an assumed MVTec leftover and **restored
in the second**, byte-identical (blob `a8fcabd`). It is not MVTec work: it copies
`PCB-cropped/all` → `PCB-cropped-combined`, i.e. it is the PCB half of the
PCB-plus-MVTec combination experiment, and is ICCE-TW-era provenance. It is
tracked, and §2.1's entry for it stands unqualified.

**Line endings normalised.** The 2024-era sources were committed with CRLF while
every 2026 ICETA-era script is LF. Seven tracked files — `dataset.py`,
`extract_code.py`, `pixelsnail.py`, `train_pixelsnail.py`, `train_vqvae.py`,
`train_vqvae-trainVal-allTransformations.py`, `vqvae.py` — were converted to LF.
The change is whitespace-only (`git diff -w` empty for each; each byte-identical
to its predecessor after stripping CR) and all still load their published
checkpoints at `strict=True`. `.gitattributes` now pins the convention, with an
explicit `results/*.csv -text` exception: those manifests are CRLF by construction
(Python's `csv` writer with `newline=''`) and their exact bytes back the split and
subsample provenance, so they are never renormalised. Verified: every
`results/*.csv` blob hash is unchanged under the new attributes.

This normalisation does **not** touch anything §2 relies on. `icce-tw-2026`
(`ef5f67c`) still carries `pixelsnail.py` at 431 CRLF lines, so the byte-level
comparisons in §2.2 against upstream remain reproducible against that tag.

**Files added.** `AUDIT.md` (independent adversarial verification of Table 1, the
10-seed paired test, the leakage controls, the synthetic-pool counts and the
filter identity), `REPRODUCE.md` (each claim → the command that regenerates it,
plus the limits on exact reproduction), `.gitattributes`, and
`results/overfit_matrix.json`.

**A gap in §2.4's spirit, now closed.** `test_overfit_matrix.py` previously
persisted only a PNG that records no metrics, so its result could not be
recovered without re-running it — the same failure mode §2.4 documents for the
ICCE-TW results. It now writes `results/overfit_matrix.json`. The committed file
is from a 2026-07-28 run, not the 2026-07-13 original, whose numbers were never
recorded anywhere and are unrecoverable.

**Overtaken statement.** The header says the tier-1 source state was committed
"as the single commit tagged `iceta-2026-submission`". That remains true of
2026-07-27; as of 2026-07-28 the tag points at the later cleanup commit and the
submission state spans three commits. The header's convention of referring to
tags by name rather than by hash is what keeps the rest of this document valid
across that move.

### 5.2 Pre-publication fixes — 2026-10-02

One commit on `master` after `iceta-2026-submission`. **The tag was not moved**: it
still marks the state submitted on 2026-07-30, which therefore still contains the
figure and code paths corrected here.

**Fig 11 regenerated; it plotted the retracted claim.** `plot_budget_abc.py`
drew `abc_recon.json`'s `B_pool` (≈0.910 at 10%) under the label "generator
trained on the FULL pool (LEAKY)". That arm has no generator (AUDIT.md §7.1); the
true full-pool generator gives 0.424 at 10%, +0.002 over its own A. The script
now reads `results/abc_budget_b*.json` (budget-restricted generator) and
`results/abc_leaky_b{10,25,50}.json` (full-pool generator), and plots B − A and
C − A for each arm against **its own** Condition-A run, with the ±0.015 rerun
band of AUDIT.md §7.5. It no longer reads `abc_recon.json`. This overtakes the
§3 table row "Fig 11 (honest vs leaky) … reads `results/abc_recon.json`,
`abc_budget_b*.json`": it now reads `abc_budget_b*.json` and `abc_leaky_b*.json`.
The §3 row "Finding 5c (leakage ceiling)" is likewise better read as
"reconstruction ceiling".

**Other places that reported +0.510 as a full-pool generator result.**
`abc_b10_seeds.py` printed "the LEAKY protocol (generator trained on the full
pool) gives +0.510" as a hard-coded constant; it now reads the full-pool
generator (+0.002) and the reconstruction ceiling (+0.510) from their result
files and labels each. `plot_abc.py` (Fig 10) labelled the reconstruction arms
"LEAKY" and their gap "leakage"; relabelled "ceiling", data unchanged.
`status.sh` echoed "leak +0.51"; relabelled. `findings.md` Finding 5c and the
matching answer were rewritten to the three-arm framing of AUDIT.md §7.6, which
is the framing the paper uses. `run_binary_track.sh` and `abc_binary.py` stated
the +0.51-as-leakage hypothesis before the experiment; that text is kept and an
OUTCOME note appended to each.

**Dead guards in `train_binary.py`.** The board-leak loop had a bare `pass` and
the train/test check ended `or True`, so neither could fire. Replaced with an
assertion that no train row is on a test board and one that no crop is in both
train and test. Both pass on the committed manifests (0 violations each) and
both were confirmed to fire on injected violations, so no binary-track result is
affected.

**AUDIT.md line references.** AUDIT.md cites `plot_budget_abc.py` and
`abc_b10_seeds.py` by line number. Those references are to the files at
`iceta-2026-submission` and remain valid there; they do not match `master`.

### 5.3 Publication — 2026-10-02

A second commit after §5.2, so §5.2's "one commit" now reads "two commits"; the tag
is still not moved.

**Untracked: `review.html`, `duplicates_review.html`, `generation_review.html`,
`samples_review.html`.** Their ~1,150 images (`*review_assets/`) were never in git,
so every clone rendered them broken. They remain on disk, regenerable by
`build_*review.py`, and are now ignored. `status.sh` still names them; it describes
the local working tree.

**README corrected.** It referred to a checkpoint release and a DOI; neither exists,
and both references are removed. It said the repository "redistributes only derived
crops"; it redistributes none, beyond the example crops shown in figures. It now also
states that development was assisted by AI coding tools.

### 5.4 The 600 px crops are the public release, byte for byte — 2026-10-02

**Finding.** `PCB-cropped/all` is not the output of a lost cropping script. It is a
verbatim copy of `JPEGImages/` from the TDD-net [2] VOC release (`VOC_PCB.zip`,
1 185 147 187 bytes, MD5 `033ce3f2ca5cdf2e20eb65dfa50bca81`), which is itself derived
from HRIPCB [4]. ([2], [4] are the ICETA paper's references: Ding et al., TDD-net, 2019;
Huang & Wei, arXiv:1901.08204.) Evidence, every file hashed, archive read in place:

| check | result |
|---|---|
| file count | archive `JPEGImages/` 10 668, `PCB-cropped/all` 10 668 |
| filenames | 10 668 common; 0 only in the archive; 0 only local |
| MD5 | **10 668 / 10 668 byte-identical**, 0 differ. By variant: plain, `l`, `rot90`, `rot270` each 2 667 / 2 667 |
| timestamps | every local file keeps the archive's own entry mtimes (2019-04-06 20:18–20:2x), which extraction preserves; `PCB-cropped/all` itself dates from 2024-12-01 |

So the 600 px crops, and also their four-way flip/rotation augmentation, came from the
release; nothing was cut or augmented locally. The release's own crop geometry is what
Finding 4 measures. The archive's provenance as *the* public release is taken from where
it was downloaded. No published hash exists to check it against.

`VOC_PCB/` was checked the same way: all 21 342 archive files (images, annotations,
`ImageSets/`, `check_xml.py`, `generate_txt.py`) are present and byte-identical, with no
extra local files.

**Consequence: the gap is closed.** The 600 px baseline, and with it the whole ICETA chain,
is reproducible from public data: download the release and use `JPEGImages/` as
`PCB-cropped/all`. §3.4's "the only break" no longer exists. The ICCE-TW 600 px crops are
reproducible in the same way.

**Overtaken statements** (left intact above, per §5's convention):
- §2.5 "**Not reproducible from this repo.** … the script that produced these 600 px crops
  ran on another machine": the files are the release's; no local script produced them.
- §2.6 row "600 px crops | **No** — cropping script ran elsewhere": now **Yes**, from the
  public release.
- §3.2 row `VOC_PCB/` "HRIPCB, as distributed": more precisely, **the TDD-net [2] VOC
  release, derived from HRIPCB [4]** (600 px crops of HRIPCB boards plus augmentation).
- §3.2 row `PCB-cropped/all` "**input, not reproducible here**": reproducible; a copy of
  `VOC_PCB/JPEGImages/`.
- §3.4 row "600 px crops | **No** — upstream input, cropping script absent", and the
  paragraph after it calling this link "the only break": now **Yes**; no break remains.
- §4 item 4 "The 600 px cropping procedure. Ran on another machine": the cropping was
  done by the release's authors and is described in [2]; nothing of ours is lost.

The same claim appears in the messages of tags `icce-tw-2026` and
`iceta-2026-submission`. Tags are not moved or rewritten (§5.2), so those messages stand
as written; this section supersedes them. `README.md`, `REPRODUCE.md` (§0, §6.2) and
`findings.md` (Finding 4) are corrected in place, each marked with a pointer here.

### 5.5 Repository layout — 2026-10-02

The 75 scripts formerly at the repository root were moved with `git mv` into four
directories by role. **No file was renamed**: each keeps its basename, so a root-level
name anywhere in this document, in AUDIT.md, or in a tag or commit message resolves
through the table below. `git log --follow <new path>` traces each file's history
across the move (every rename is detected; minimum similarity 83%). Both tags still
point at the old flat layout.

| new directory | files | moved from the root (old path = basename) |
|---|---|---|
| `src/` | 20 | `build_split.py`, `dataset.py`, `extract_code.py`, `extract_code_labeled.py`, `make_budget_subsamples.py`, `make_tight_crops.py`, `pcb_utils.py`, `pixelsnail.py`, `sample.py`, `scheduler.py`, `train_binary.py`, `train_classifier.py`, `train_pixelsnail.py`, `train_pixelsnail_cond.py`, `train_vqvae.py`, `train_vqvae_tight.py`, `vqvae.py` and `distributed/` (whole package) |
| `experiments/` | 29 | `abc_b10_seeds.py`, `abc_binary.py`, `abc_budget.py`, `abc_recon.py`, `calibrate_sliding.py`, `cond_check.py`, `cond_check_n360_report.py`, `detector_600.py`, `eval_prior_nll.py`, `gate_check.py`, `make_defect_detector.py`, `run_b10_seeds.sh`, `run_binary_track.sh`, `run_budget_b25_b50.sh`, `run_budget_pipeline.sh`, `run_budget_pipeline_resume.sh`, `run_cond_check_n360.sh`, `run_cond_checks.sh`, `run_converge.sh`, `run_genreview.sh`, `run_nll_eval.sh`, `run_reorder.sh`, `run_samples_review.sh`, `sample_cond.py`, `sample_pool.py`, `sample_uncond.py`, `score_pools_600.py`, `test_conditioning.py`, `test_overfit_matrix.py` |
| `analysis/` | 19 | `build_duplicates_review.py`, `build_generation_review.py`, `build_review.py`, `build_samples_review.py`, `checkpoint_inventory.py`, `find_duplicates.py`, `make_fig_samples.py`, `make_negative_figs.py`, `make_paper_figs.py`, `nn_baseline.py`, `nn_baseline_600.py`, `nn_pixel.py`, `plot_abc.py`, `plot_budget_abc.py`, `plot_consistency.py`, `plot_loss_curves.py`, `plot_nll.py`, `status.sh`, `status_checks.py` |
| `legacy/` | 7 | `augmentGoodPCBImages.py`, `demo_app.py`, `DEMO_README.md`, `generate_pool.py`, `make_timelapse.py`, `randomlyCopyImageFiles.py`, `train_vqvae-trainVal-allTransformations.py` |

Kept at the root: `README.md`, `PROVENANCE.md`, `REPRODUCE.md`, `AUDIT.md`,
`findings.md`, `LICENSE`, `.gitignore`, `.gitattributes`, `results/`, `figures/`,
`fig_samples.{png,pdf}` and `fig_samples_NOTE.txt`.

**Why module names stay flat.** Every row in every `lmdb/` store is a pickled
`dataset.CodeRow`, so `dataset` must stay importable under that bare name. Turning the
directories into packages (`src.dataset`) would have made all eight existing stores
unreadable. Instead, the 28 scripts that import a module from another directory begin
with a two-line `sys.path` bootstrap that adds `src/`, `experiments/` and `analysis/`.
Imports are otherwise unchanged. Scripts are run from the repository root as before:
`python experiments/abc_budget.py`.

**Other code changes, all path-only.**
- The `run_*.sh` drivers and `status.sh` now invoke scripts by their new paths.
- `analysis/make_fig_samples.py` resolves the repository root one directory up.
- `analysis/build_review.py` reads `src/make_tight_crops.py`.
- Usage strings in `legacy/` now name their new paths.

Line numbers shift by two in the 28 bootstrapped files. AUDIT.md's line references were
already pinned to `iceta-2026-submission` (§5.2).

**Running AUDIT.md's commands on master.** Its inline `python3 -c "from pcb_utils import
…"` blocks need `PYTHONPATH=src:experiments:analysis`. All four such blocks were checked:
they fail without it and pass with it. Its `sed -n 'N,Mp' <script>` excerpts refer to the
tagged files; run them at the tag.

**Verified.**
- Every moved module's top-level imports were executed under the exact `sys.path` that
  running it would produce, and each local import resolved to its new path.
- One row was read from every `lmdb/` store through `src/dataset.py`, under an
  `experiments/` script's path setup.
- `analysis/plot_budget_abc.py`, `analysis/plot_abc.py` and
  `analysis/make_fig_samples.py` regenerated their committed PNGs and
  `fig_samples_NOTE.txt` byte-identical. The PDF differed only in Matplotlib version and
  creation-date metadata, and was not recommitted.
- Thirteen argparse entry points ran `--help` from an unrelated working directory.
- All 13 shell scripts pass `bash -n`.
- No moved script is still invoked or backtick-referenced by its root path in
  README.md, REPRODUCE.md, findings.md or `legacy/DEMO_README.md`.

**Known wart, recorded rather than fixed.** `src/train_binary.py` imports
`experiments/abc_recon.py` (for `train_clf` and the transforms), so `src/` is not free
of `experiments/`. It worked unchanged before the move and works the same after it.

### 5.6 Camera-ready paper tracked — 2026-10-03

`paper/ICETA-2026.pdf` is the final ICETA 2026 camera-ready (2 pages, MD5
`4c31182b34c5a79775eacee5c92c7d39`). It overtakes the statement in the
`iceta-2026-submission` tag message that the manuscript is not in this repository.
That tag still marks the submission state, not this version.

**Reference numbers changed.** §5.4 cites the paper's references as TDD-net [2] and
HRIPCB [4], which was their numbering in the draft current on 2026-10-02. In the
camera-ready, **TDD-net (Ding et al., 2019) is [5]**; HRIPCB (Huang & Wei,
arXiv:1901.08204) is still [4]; and [2] is now the ICCE-TW paper. Read §5.4's "[2]" as
[5]. The camera-ready's wording for the data, "from the released 600 × 600 crops [5]",
is the one §5.4's evidence supports.
