# Findings — ICETA 2026

**Headline:** Crop methodology, not generative AI, is what moves PCB defect classification.
A size-matched change of crop scale is worth **+0.65 macro-F1**; the class-conditional
generator the ICCE-TW paper advocates produces **no usable labeled synthetic defects**.

Figures: `figures/fig1–fig6`, `fig11_budget_abc` (budget-restricted vs full-pool generator). Numbers:
`results/paper_tables.md`, `results/abc_budget_b{10,25,50,100}.json`,
`results/cond_check_ep320_n360_pooled.json`.

---

## Method (leakage-controlled)
- **Board-level split.** 10 HRIPCB boards; boards **06 + 09 held out** as test, touched by
  nothing — not the classifier, not the VQ-VAE, not the priors. All 4 augmentation variants
  of a crop share a board, so the board split has no augmentation leakage.
- **Classifier.** ResNet-18 (ImageNet-pretrained), 6-way, an *identical* recipe everywhere
  (224 input, flips/rotations, Adam 1e-4, 20 epochs, batch 64). Metric: **macro-F1** on the
  held-out boards, mean ± std over **3 seeds**. Chance = 0.167.
- **Only variable = crop scale.** Same defects, same boards, same recipe.
  - *600px*: the original crops (defect somewhere in a 600×600 frame).
  - *TIGHT*: 128px native window centred on the annotated bbox → resized to 256.
  - *matched* (2149 train, one crop per base crop) isolates **scale**; *per-bbox* (4274 train)
    adds the **size** effect.

---

## Finding 1 — Crop scale dominates (Fig 1a, Fig 3; Table 1)
At 100% budget, **size-matched** (both 2149 training crops):

| set | n_train | macro-F1 | accuracy |
|---|---|---|---|
| 600px (baseline) | 2149 | **0.246 ± 0.013** | 0.315 |
| TIGHT matched | 2149 | **0.897 ± 0.020** | 0.896 |
| TIGHT per-bbox | 4274 | 0.947 ± 0.005 | 0.947 |

- **Crop scale: +0.65 macro-F1 (+265%)**, with dataset size held fixed.
- **Dataset size: +0.05** (doubling the data). ⇒ **~93% of the gain is crop scale, ~7% is size.**
  The effect is *not* confounded by dataset size.

## Finding 2 — It repairs every collapsed class (Fig 1b; Table 3)
| class | 600px | tight |
|---|---|---|
| missing_hole | 0.78 | 1.00 |
| mouse_bite | 0.25 | 0.85 |
| open_circuit | 0.14 | 0.89 |
| short | 0.19 | 0.94 |
| spur | **0.00** | **0.90** |
| spurious_copper | 0.13 | 0.81 |

Cause: the median defect is **27×27 px inside a 600×600 crop** (~4.5% of frame). After the
224 resize it becomes a ~10px needle the classifier cannot localize.

## Finding 3 — Scarcity behaviour (Fig 2; Table 2)
| budget | 600px | TIGHT |
|---|---|---|
| 10% | 0.144 ± 0.035 | **0.422 ± 0.036** |
| 25% | 0.191 ± 0.029 | **0.640 ± 0.010** |
| 50% | 0.252 ± 0.016 | **0.779 ± 0.007** |
| 100% | 0.246 ± 0.013 | **0.897 ± 0.020** |

The 600px baseline is *at chance* at the 10% budget — the very regime the scarcity argument
rests on. With correct crops, a plain ResNet-18 reaches **0.42 at 10%** and **0.90 at 100%**
from ~115 real crops/class, **before any synthetic data**.

## Finding 4 — The original crops were never defect-centred (Fig 4)
Recovered from the VOC bbox annotations (the original cropping script ran on another machine):
defect positions inside the 600px crops are **effectively uniform-random** — median **173px**
from centre, only **5.5%** within 50px, position std ≈160px vs ~166px for uniform placement.
The dataset's own `generate_txt.py` also performs a **crop-level random split**, i.e. the
board-recognition leakage our board-level split exists to prevent.

---

## Finding 5 — The generative arm fails, in an instructive way (Fig 5, Fig 6, Fig 11)
**Setup.** VQ-VAE-2 retrained on tight *train-pool* crops (4274; test boards excluded → the
generator never sees them), final recon MSE **0.00069**. Class-conditional PixelSNAIL priors
(top + bottom, class injected as feature bias + per-block condition + output-logit bias)
trained on the tight codes. Conditioning check: sample per class → decode → classify with the
ResNet-18 that reads **real crops at 0.926** and **VQ-VAE reconstructions at 0.918** (so the
judge and the decoder are not the bottleneck).

**5a. Conditioning develops a real signal with training — but never de-collapses.**
At 80 epochs, conditioning consistency was 22.4% ≈ chance (16.7%). Training to 320 epochs
grew a genuine label→output association: Cramér's V rose monotonically 0.216 → 0.376 across
five checks. An n=72 check at epoch 320 passed our three pre-registered criteria — but an
audit-triggered replication at **n=360** showed that pass was sampling noise on the collapse
criterion:

| statistic (n=360, ±SE) | value | pre-registered criterion | verdict |
|---|---|---|---|
| consistency | 37.8% ± 2.6 | (chance 16.7%) | — |
| collapse index | **0.483 ± 0.026** (spur) | ≤ 0.40 | **fail, >3 SE** |
| consistency ex-collapse | 30.3% ± 2.7 | ≥ 0.30 | marginal pass |
| Cramér's V | 0.376 ± 0.025 | ≥ 0.30 | pass |

The label does something real (V is 15 SE above zero) — but 48% of everything the sampler
draws classifies as *spur*, whatever was asked for. A clean from-scratch full-data prior
(no warm-start) reproduces the same picture (37.5%, V 0.473, collapse 0.458), so this is a
property of the data regime, not of any initialization. Conditioning strength vs generator
data (from-scratch priors, n=72 each): b10 15.3% (V 0.24) → b25 50.0% (V 0.50) → b50 52.8%
(V 0.57) → b100 37.5% (V 0.47) — every budget stays at or above the 0.40 collapse line.

**5b. Downstream, honestly measured, synthetic data does not help — it mostly hurts.**
A/B/C at every budget (A = real only; B = A + 360 synthetic; C = A + filter-kept synthetic;
filter = the budget-b Condition-A model, no leakage; generator, VQ-VAE and priors trained
from scratch on ONLY that budget's subsample; macro-F1, 3 seeds):

| budget | A (real) | B (+synthetic) | C (+filtered) | filter keeps |
|---|---|---|---|---|
| 10% | 0.418 ± 0.011 | 0.431 ± 0.029 | 0.433 ± 0.017 | 20–27% |
| 25% | 0.641 ± 0.005 | 0.595 ± 0.009 | 0.632 ± 0.025 | 54–56% |
| 50% | 0.804 ± 0.001 | 0.759 ± 0.014 | 0.814 ± 0.007 | 49–51% |
| 100% | 0.899 ± 0.019 | 0.848 ± 0.031 | 0.884 ± 0.014 | 28–32% |

Unfiltered synthetic data **degrades** the classifier by ~0.05 macro-F1 at every budget
above 10% — the pools are majority-mislabeled (5a), so Condition B is a label-noise
injection. The no-leakage filter functions as damage control: it discards half to four
fifths of the pool and returns performance to baseline, **never meaningfully above it**
(best case +0.010 at 50%, within seed noise).

**5c. The generator, not its data, is the bottleneck.** *(Revised 2026-10-02. The earlier
text called +0.51 "leakage" a full-pool generator would report; AUDIT.md §7 shows that number
involves no generator, and the claim is retracted.)* Three arms at the 10% budget, each
against its own Condition A:

| arm | what is added | B − A at 10% |
|---|---|---|
| budget-restricted generator | 360 samples, generator trained on the 10% subsample | +0.012 |
| full-pool generator | 360 samples, generator trained on all 2,149 train crops | **+0.002** |
| reconstruction ceiling | VQ-VAE round-trips of all 2,149 real crops, no generator | +0.510 |

A generator that saw the whole training set buys nothing over one that saw 10% of it (and at
25%/50% it hurts: −0.066 / −0.017), while merely round-tripping that training set through the
same decoder buys +0.51. The failure is in generation, not in what the generator was allowed
to see. Sources: `results/abc_budget_b*.json`, `results/abc_leaky_b*.json`,
`results/abc_recon.json`; figures `fig11_budget_abc`, `fig10_abc_recon`.

**Why it fails.** Not the autoencoder (recon classifies at 0.918), not plumbing (the label
flips 70% of top-code predictions; teacher-forced code accuracy improved 0.41 → 0.61 with
stronger conditioning). The ~27px defect is ~1% of the crop's pixels, so the prior's
likelihood is dominated by substrate; free-running sampling drifts to the generic-PCB mode
and rarely commits to the class-specific defect. The same pixel-budget asymmetry that broke
the 600px classifier (Finding 2) breaks the generator — crop scale is the common cause.

**Scope.** This is a negative result for *this* architecture and data regime (VQ-VAE-2 +
class-conditional PixelSNAIL, ~700 real crops/class, 27px defects). It does not prove
generative augmentation is impossible for PCB inspection. It does show that the pipeline
advocated by the ICCE-TW paper, as configured, **yields no usable labeled synthetic defects**
— while the far cheaper fix (crop the defect properly) delivered +0.65 macro-F1.

## Finding 6 — Label-free synthetic positives don't help either (binary track)
The 6-way failure could be blamed on labels: conditioning collapses, so pools are
mislabeled. The binary track removes that excuse. Task: **defect vs no-defect** — what an
AOI line actually needs. Positives = the tight crops; negatives = windows from the same
600px crops, same boards, same 128px→256 recipe, rejection-sampled to miss every bbox
(the only difference is the defect). Synthetic positives come from *unconditional* priors
trained only on defect crops — **every sample is a positive by construction, so no label
can be wrong.** Same board split, same recipe, 3 seeds; honest generators see only their
budget's subsample, the leaky generator sees the whole train pool.

| budget | A (real) | B honest | C honest | B leaky | C leaky |
|---|---|---|---|---|---|
| 10% | 0.841 ± 0.006 | 0.859 ± 0.021 | 0.858 ± 0.027 | 0.843 ± 0.015 | 0.844 ± 0.009 |
| 25% | 0.925 ± 0.011 | 0.932 ± 0.008 | 0.910 ± 0.019 | 0.940 ± 0.002 | 0.937 ± 0.012 |
| 50% | 0.960 ± 0.002 | 0.962 ± 0.002 | 0.960 ± 0.008 | 0.963 ± 0.003 | 0.958 ± 0.006 |
| 100% | 0.980 ± 0.005 | 0.981 ± 0.003 | 0.975 ± 0.001 | 0.972 ± 0.009 | 0.978 ± 0.007 |

Best case: **+0.018 at the 10% budget, inside seed noise** (seed spread ±0.021); < +0.008
everywhere else; even the *leaky* pool moves nothing (+0.002 at 10%). Two diagnostics say
why. First, the baseline has little room: with correct crops, 854 real images already give
0.841. Second, **the filter rejects 64–95% of the full-pool generator's samples** — positives
by construction that the budget-b detector calls clean board — so for the leaky pool the
failure is demonstrably in the *content*: the unconditional prior renders substrate, not
defects. The honest generators' samples pass the filter far more often (keep rates 21–91%,
consistent with memorizing their own small subsample), yet still add at most +0.018 —
accepted or rejected, the pools contain nothing the detector can use. This closes the last
escape route for the generative thesis in this regime: with labels guaranteed correct,
synthetic positives still do not help.

---

## Answers to the questions posed
- **Does synthetic data help?** No, on both tasks. 6-way: unfiltered synthetic data *hurts*
  by ~0.05 macro-F1 everywhere above the 10% budget, and the filtered condition never
  meaningfully beats real-only (best +0.010, within noise). Binary — where labels are
  correct by construction — best case +0.018 at 10%, inside seed noise, and even the leaky
  pool moves nothing.
- **Does the benefit grow as data gets scarcer?** The opposite direction is true twice over:
  at low budgets the honest generator conditions *worse* (15.3% at b10 — chance), and a
  generator trained on the full pool does no better (+0.002 at 10%); the only large gain at
  10% (+0.51) comes from reconstructions of real crops, with no generator involved (Finding 5c). For **crop scale**, the benefit is large at every budget and the
  corrected baseline retains real signal even at 10% (0.42 vs 0.14).
- **What fraction of generated images is unusable?** As labeled data: 62–85% carry the wrong
  class depending on budget (conditioning consistency 15–53%); the no-leakage filter itself
  rejects 44–80% of every pool. What survives adds nothing measurable.
- **Anything contradicting the ICCE-TW qualitative claims?** Yes, and we state it: the defect-
  scarcity difficulty that motivates generative AI is substantially an artifact of crop
  methodology; the generative pipeline delivers no usable labeled defects; and the standard
  evaluation design (full-data generator, subsampled classifier) manufactures the very
  benefit it claims to test.
