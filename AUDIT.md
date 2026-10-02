# AUDIT — adversarial verification of five claims

> **Paths (added 2026-10-02).** This audit was written when every script sat at the repository
> root; its file and line references match the files at tag `iceta-2026-submission`. Scripts
> now live in `src/`, `experiments/`, `analysis/` and `legacy/` under the same basenames
> (mapping: PROVENANCE.md §5.5). On master, run the `python3 -c` blocks below with
> `PYTHONPATH=src:experiments:analysis`, and the `sed -n` excerpts at the tag.

> **Scope note (appended 2026-07-28).** The title records this document's original commission —
> five scoped claims, §1–§5. Scope has since grown to **31 claims across §1–§6**: the original
> five, plus the 26 remaining ICETA numbers verified in §6. The title is left as first written;
> this document is amended by appending, never by editing an earlier statement.

Date: 2026-07-27. Auditor operated under: *trust no summary document, no memory file, no
`findings.md`, no `PROVENANCE.md`*. Every number below was read out of a raw result JSON, a
raw CSV, an LMDB code store, a `.pt` checkpoint's embedded `args`, or a run log. Summary
documents were consulted **only** to locate which raw field a claim refers to, never as
evidence. No code was changed.

Scope: exactly the five claims requested. Nothing else in the repo is endorsed by this file.

**Verdicts: 5 PASS, 0 FAIL, 0 MISMATCH.** Two provenance caveats are recorded (§3.4, §4.2);
neither invalidates a published number, and both are stated in full below.

---

## Summary table

| # | Claim | Verdict |
|---|---|---|
| 1 | Table 1's twelve A/B/C numbers + four leaky numbers | **PASS** (16/16 exact to 3 dp) |
| 2 | 10-seed paired test: B−A = −0.008, CI [−0.031,+0.014], t(9) = −0.81, p = 0.44 | **PASS** (4/4 exact) |
| 3 | No leakage of test boards 06/09 into any generator or classifier, incl. warm-start lineage | **PASS** (with caveat §3.4) |
| 4 | Non-empty synthetic pools for every A/B/C run | **PASS** (360 files/budget, 60/class, all distinct; caveat §4.2) |
| 5 | Filter at budget b = the Condition-A classifier at budget b | **PASS** (same Python object, same seed) |

---

## Claim 1 — The twelve A/B/C numbers and the four leaky numbers

### 1.1 The twelve honest A/B/C numbers

**Source files:** `results/abc_budget_b10.json`, `..._b25.json`, `..._b50.json`, `..._b100.json`
(written by `abc_budget.py:106-107`; each is a JSON list of 3 records, one per classifier seed).

**Exact field:** `[i]["A"|"B"|"C"]["macro_f1"]`, arithmetic mean over the 3 seed records,
rounded to 3 dp. Aggregation is `np.mean` over seeds — confirmed at `plot_budget_abc.py:34-35`.

**Command:**

```bash
cd /mnt/storage/PycharmProjects/vq-vae-2-pytorch
python3 -c "
import json, statistics as st
for b in [10,25,50,100]:
    d=json.load(open(f'results/abc_budget_b{b}.json'))
    print(b, {k: round(st.mean([r[k]['macro_f1'] for r in d]),3) for k in 'ABC'})
"
```

**Actual on-disk values (mean over 3 seeds, and the per-seed values they came from):**

| budget | cond | claimed | **actual (6 dp)** | rounded | per-seed macro_f1 (seed 0, 1, 2) | verdict |
|---|---|---|---|---|---|---|
| 10% | A | 0.419 | **0.418598** | 0.419 | 0.405454, 0.419152, 0.431188 | PASS |
| 10% | B | 0.431 | **0.430876** | 0.431 | 0.467202, 0.395977, 0.429448 | PASS |
| 10% | C | 0.433 | **0.433265** | 0.433 | 0.450416, 0.409709, 0.439671 | PASS |
| 25% | A | 0.641 | **0.641112** | 0.641 | 0.647496, 0.639309, 0.636532 | PASS |
| 25% | B | 0.595 | **0.595425** | 0.595 | 0.606565, 0.583453, 0.596256 | PASS |
| 25% | C | 0.632 | **0.631747** | 0.632 | 0.635256, 0.598999, 0.660986 | PASS |
| 50% | A | 0.804 | **0.803530** | 0.804 | 0.805314, 0.802230, 0.803046 | PASS |
| 50% | B | 0.759 | **0.759317** | 0.759 | 0.739032, 0.767616, 0.771302 | PASS |
| 50% | C | 0.814 | **0.814324** | 0.814 | 0.821683, 0.816392, 0.804896 | PASS |
| 100% | A | 0.899 | **0.899280** | 0.899 | 0.917887, 0.872692, 0.907262 | PASS |
| 100% | B | 0.848 | **0.847750** | 0.848 | 0.825049, 0.827186, 0.891014 | PASS |
| 100% | C | 0.884 | **0.884422** | 0.884 | 0.866954, 0.902000, 0.884311 | PASS |

**12/12 PASS.** No number required rounding closer than 0.0006 from its printed value.

Metric definition audited, not assumed: `macro_f1` is unweighted mean of the six per-class F1s
computed on the held-out board test set (`train_classifier.py:71-83`), evaluated **once after
the final epoch** with no best-epoch/test-set model selection (`abc_recon.py:99-106`,
`train_clf` evaluates after the epoch loop terminates). The separate `best_macro_f1` field that
exists in `results/classifier_*.json` is **not** used by any of these twelve numbers.

### 1.2 The four leaky numbers

**Source file:** `results/abc_recon.json` (written by `abc_recon.py:170-171`; 12 records =
4 budgets × 3 seeds). This is a *different experiment* from `abc_budget_*`: synthetic data is
replaced by VQ-VAE reconstructions of the **full** train pool.

**Exact field:** `[i]["B_pool"]["macro_f1"]`, mean over the 3 records at each `["budget"]`
value (0.1 / 0.25 / 0.5 / 1.0). Selector confirmed at `plot_budget_abc.py:44-46`.

**Command:**

```bash
python3 -c "
import json, statistics as st
R=json.load(open('results/abc_recon.json'))
for b in [0.1,0.25,0.5,1.0]:
    v=[r['B_pool']['macro_f1'] for r in R if abs(r['budget']-b)<1e-9]
    print(b, round(st.mean(v),3), [round(x,6) for x in v])
"
```

**Actual on-disk values:**

| budget | claimed | **actual (6 dp)** | rounded | per-seed | verdict |
|---|---|---|---|---|---|
| 10% | 0.910 | **0.909555** | 0.910 | 0.927463, 0.903771, 0.897430 | PASS |
| 25% | 0.911 | **0.910552** | 0.911 | 0.906850, 0.918433, 0.906373 | PASS |
| 50% | 0.925 | **0.925117** | 0.925 | 0.927229, 0.925040, 0.923081 | PASS |
| 100% | 0.928 | **0.927547** | 0.928 | 0.919900, 0.938360, 0.924380 | PASS |

**4/4 PASS.**

Supporting context read from the same file (not claimed, but needed to confirm the numbers
mean what the label "leaky" says): the comparison baseline `A` in *this* file is
0.399530 / 0.645780 / 0.803956 / 0.917775, so the 10% "+0.51 lift" is 0.909555 − 0.399530 =
+0.510. The leaky training-set sizes are `B_pool.n_train` = 2364 / 2686 / 3224 / 4298, i.e.
`n_real` (215/537/1075/2149) **+ 2149 reconstructions of the whole train pool** at every
budget — which is exactly the leak the label asserts.

### 1.3 Independent cross-check against the run log

The pipeline's own summary print (`plot_budget_abc.py:100-105`), captured at
`logs/budget_b25_b50.log:11-15`, was written by a *separate* process from the JSONs and agrees
with all sixteen recomputed values:

```
 budget                A         B honest         C honest     B_pool leaky   keep
    10%     0.419±0.011     0.431±0.029     0.433±0.017      0.910±0.013    23%
    25%     0.641±0.005     0.595±0.009     0.632±0.025      0.911±0.006    55%
    50%     0.804±0.001     0.759±0.014     0.814±0.007      0.925±0.002    50%
   100%     0.899±0.019     0.848±0.031     0.884±0.014      0.928±0.008    30%
```

Command: `sed -n '11,15p' logs/budget_b25_b50.log`

---

## Claim 2 — 10-seed paired test, recomputed from per-seed values

**Source file:** `results/abc_budget_b10_seeds.json` — a JSON list of **10** records
(`seed` 0…9), written incrementally by `abc_b10_seeds.py`.

**Exact fields:** `[i]["A"]["macro_f1"]` and `[i]["B"]["macro_f1"]`. Nothing else was used;
the file's own printed statistics were ignored and the test was recomputed from scratch.

**Command (recomputes t, CI and p from the raw per-seed values):**

```bash
python3 -c "
import json, math, statistics as st
from scipy import stats
d=json.load(open('results/abc_budget_b10_seeds.json'))
A=[r['A']['macro_f1'] for r in d]; B=[r['B']['macro_f1'] for r in d]
diff=[b-a for a,b in zip(A,B)]; n=len(diff)
m,sd = st.mean(diff), st.stdev(diff); se = sd/math.sqrt(n); t = m/se
tc = stats.t.ppf(0.975, n-1); p = 2*stats.t.sf(abs(t), n-1)
print('n=%d  mean=%.6f  sd=%.6f  se=%.6f' % (n, m, sd, se))
print('t(%d)=%.4f  p=%.6f  CI=[%.6f, %.6f]' % (n-1, t, p, m-tc*se, m+tc*se))
"
```

**Actual on-disk values.** n = 10 records confirmed (`seed` 0–9, no duplicates, no gaps).

Per-seed inputs actually read from the file:

| seed | A macro_f1 | B macro_f1 | d = B − A |
|---|---|---|---|
| 0 | 0.409410 | 0.475726 | +0.066315 |
| 1 | 0.423163 | 0.405351 | −0.017812 |
| 2 | 0.433400 | 0.426927 | −0.006473 |
| 3 | 0.427613 | 0.376613 | −0.051000 |
| 4 | 0.416184 | 0.404943 | −0.011241 |
| 5 | 0.443895 | 0.453140 | +0.009245 |
| 6 | 0.463368 | 0.453796 | −0.009571 |
| 7 | 0.429477 | 0.387861 | −0.041616 |
| 8 | 0.452065 | 0.442101 | −0.009964 |
| 9 | 0.425946 | 0.417381 | −0.008565 |

| statistic | claimed | **actual** | verdict |
|---|---|---|---|
| mean B − A | −0.008 | **−0.008068** | PASS |
| 95% CI | [−0.031, +0.014] | **[−0.030582, +0.014446]** | PASS |
| t(9) | −0.81 | **−0.8107** (df = 9 confirmed, n = 10) | PASS |
| p (two-sided) | 0.44 | **0.438475** | PASS |

Supporting: sd of differences = 0.031473, se = 0.009953, t-crit(.975, df=9) = 2.262157.
Mean A = 0.432452, mean B = 0.424384, mean C = 0.429074.

**4/4 PASS.** The design is genuinely paired — verified in source, not assumed: the real
subsample is `build_train_items(rows, 1.0, seed=0)` (`abc_b10_seeds.py:53`), i.e. **frozen
across all 10 seeds**, and the synthetic pool is read once from disk with no regeneration
(`abc_b10_seeds.py:55`). Seed `s` therefore varies classifier initialisation only, which is
the precondition for the paired test being the right test.

**Scope limitation, stated because the file states it and it is true:** variance here is over
classifier initialisation only, **not** over data subsampling. This test cannot generalise
over draws of 215 crops. (`abc_b10_seeds.py:10-15`.)

---

## Claim 3 — No leakage of test boards 06/09

The test set is boards **06** and **09** — `results/splits.json` → `"test_boards": ["06","09"]`,
`"train_boards": ["01","04","05","07","08","10","11","12"]`. Board identity comes from the
filename (`pcb_utils.py:101-104`, `light_NN_<class>_..._600.jpg`), and all 4 augmentation
variants of a crop share a board, so a board split admits no augmentation leakage
(`build_split.py:34`).

This claim was verified **empirically from data artefacts**, not from the assertions in the
scripts — because an assertion only proves what ran *if* it ran.

### 3.1 Every manifest: no test board in any train split

```bash
python3 -c "
import csv
for m in ['manifest.csv','manifest_tight.csv','manifest_tight_perbbox.csv',
          'manifest_tight_b10.csv','manifest_tight_b25.csv','manifest_tight_b50.csv',
          'manifest_nodefect.csv']:
    rows=list(csv.DictReader(open('results/'+m)))
    tr={r['board_id'] for r in rows if r['split']=='train'}
    te={r['board_id'] for r in rows if r['split']=='test'}
    print(f'{m:34s} train={sorted(tr)} test={sorted(te)} LEAK={sorted(tr & {\"06\",\"09\"}) or \"none\"}')
"
```

**Actual:**

| manifest | rows | train | test | train boards | LEAK |
|---|---|---|---|---|---|
| `manifest.csv` | 10668 | 8596 | 2072 | 01,04,05,07,08,10,11,12 | **none** |
| `manifest_tight.csv` | 2667 | 2149 | 518 | 01,04,05,07,08,10,11,12 | **none** |
| `manifest_tight_perbbox.csv` | 4792 | 4274 | 518 | 01,04,05,07,08,10,11,12 | **none** |
| `manifest_tight_b10.csv` | 733 | 215 | 518 | 01,04,05,07,08,10,11,12 | **none** |
| `manifest_tight_b25.csv` | 1055 | 537 | 518 | 01,04,05,07,08,10,11,12 | **none** |
| `manifest_tight_b50.csv` | 1593 | 1075 | 518 | 01,04,05,07,08,10,11,12 | **none** |
| `manifest_nodefect.csv` | 5416 | 4274 | 1142 | 01,04,05,07,08,10,11,12 | **none** |

Test boards 06 and 09 appear **only** in `split == test` rows, in every manifest. The
classifier's test set is `split=='test' and variant=='plain'` = **518** crops, boards 06+09
only (`train_classifier.py:66-68`); the classifier's train set is
`split=='train' and variant=='plain'` (`train_classifier.py:51-63`) — so no classifier at any
budget can see a test board.

### 3.2 Every generator's actual training data: read the code stores, not the scripts

The priors are trained on LMDB code stores. Filenames are preserved inside the store
(`extract_code_labeled.py:71`), so the board of every single training example the priors ever
saw can be recovered directly. This is the decisive check.

```bash
.venv/bin/python -c "
import lmdb, pickle, sys, collections; sys.path.insert(0,'.')
from dataset import CodeRow
from pcb_utils import parse_filename
for name in ['lmdb/train_pool_tight','lmdb/b10','lmdb/b25','lmdb/b50','lmdb/train_pool','lmdb/test_tight']:
    env=lmdb.open(name, readonly=True, lock=False)
    with env.begin() as txn:
        n=int(txn.get(b'length'))
        c=collections.Counter(parse_filename(pickle.loads(txn.get(str(i).encode())).filename)['board'] for i in range(n))
    print(f'{name:24s} rows={n:5d} boards={dict(sorted(c.items()))} TEST_BOARD_ROWS={ {k:v for k,v in c.items() if k in (\"06\",\"09\")} }')
"
```

**Actual:**

| code store | rows | boards present | test-board rows | used by |
|---|---|---|---|---|
| `lmdb/b10` | 215 | 01,04,05,07,08,10,11,12 | **{} (zero)** | `prior_b10_{top,bottom}` |
| `lmdb/b25` | 537 | 01,04,05,07,08,10,11,12 | **{} (zero)** | `prior_b25_{top,bottom}` |
| `lmdb/b50` | 1075 | 01,04,05,07,08,10,11,12 | **{} (zero)** | `prior_b50_{top,bottom}` |
| `lmdb/train_pool_tight` | 4274 | 01,04,05,07,08,10,11,12 | **{} (zero)** | `prior_b100_{top,bottom}` |
| `lmdb/train_pool` | 8596 | 01,04,05,07,08,10,11,12 | **{} (zero)** | (not used by any claimed number) |
| `lmdb/test_tight` | 518 | **06, 09 only** | 254 + 264 | evaluation only — never a `--path` for any prior |

Zero test-board rows in every store that trains a published generator. `lmdb/test_tight` is
the only store containing boards 06/09 and it is never passed as `--path` to
`train_pixelsnail_cond.py` in any run (verified in §3.3 from the checkpoints themselves).

### 3.3 Checkpoint ancestry — traced from each checkpoint's own embedded `args`

`train_pixelsnail_cond.py` saves `{'model':…, 'args':…}`, so each published prior carries its
own training provenance. This is stronger than a log: it is inside the artefact.

```bash
.venv/bin/python -c "
import torch, glob, warnings; warnings.filterwarnings('ignore')
for f in sorted(glob.glob('checkpoint/prior_b*_320.pt')):
    a=torch.load(f, map_location='cpu')['args']
    print(f'{f:42s} warm={a.warm!r}  path={a.path!r}  epoch={a.epoch}')
"
```

**Actual — every published generator, warm-start field and training corpus:**

| checkpoint | `warm` | `path` | epoch |
|---|---|---|---|
| `prior_b10_top_320.pt` | **None** | `lmdb/b10` | 320 |
| `prior_b10_bottom_320.pt` | **None** | `lmdb/b10` | 320 |
| `prior_b25_top_320.pt` | **None** | `lmdb/b25` | 320 |
| `prior_b25_bottom_320.pt` | **None** | `lmdb/b25` | 320 |
| `prior_b50_top_320.pt` | **None** | `lmdb/b50` | 320 |
| `prior_b50_bottom_320.pt` | **None** | `lmdb/b50` | 320 |
| `prior_b100_top_320.pt` | **None** | `lmdb/train_pool_tight` | 320 |
| `prior_b100_bottom_320.pt` | **None** | `lmdb/train_pool_tight` | 320 |

**`warm=None` on all 8.** There is no warm-start lineage to trace: every published prior is
initialised from scratch. The known test-leaked lineage
(`pixelsnail_top_357` → `condtight_060` → `condtight2_080` → `condtight3`, where
`pixelsnail_top_357.pt` dates from 2024-12-02 and was trained on `lmdb/all` = all 10 boards)
therefore terminates at `condtight3` and cannot reach any A/B/C number. See §3.5.

Decoders (VQ-VAEs), traced the same way:

| budget | decoder used at sampling | trained on | evidence |
|---|---|---|---|
| 10% | `checkpoint/vqvae_b10_560.pt` | `manifest_tight_b10.csv` split=train (215) | `logs/budget_pipeline.log:17` (`Namespace(manifest='results/manifest_tight_b10.csv', split='train', …)`) |
| 25% | `checkpoint/vqvae_b25_560.pt` | `manifest_tight_b25.csv` split=train (537) | `logs/budget_pipeline.log:2198` |
| 50% | `checkpoint/vqvae_b50_560.pt` | `manifest_tight_b50.csv` split=train (1075) | `logs/budget_pipeline.log:3474` |
| 100% | `checkpoint/vqvae_tight_560.pt` | `manifest_tight_perbbox.csv` split=train (4274) | `logs/rebase_tight_5090.log:2-3` |

`train_vqvae_tight.py:72` constructs `VQVAE()` fresh with no state-dict load — **a VQ-VAE
warm-start is not expressible in this script**, so no decoder can inherit test-board weights
either. `logs/rebase_tight_5090.log:3` prints
`VQ-VAE tight: training on 4274 crops (test boards ['06', '09'] excluded)`.

`lmdb/train_pool_tight` was encoded by `vqvae_tight_560` —
`logs/rebase_tight_5090.log:565-567`: `===== [2/4] extract tight codes with vqvae_tight_560 =====`
/ `encoding 4274 train crops -> lmdb/train_pool_tight` / `done, wrote 4274 code rows`.

Command: `grep -n -m4 -E "Namespace|VQ-VAE tight: training on" logs/rebase_tight_5090.log; grep -n "Namespace" logs/budget_pipeline.log`

### 3.4 CAVEAT (recorded, not a leak): at b=100 the generator corpus is a superset of the classifier corpus

`abc_budget.py:3-4` states "Everything the generator saw at budget b is exactly what the
classifier gets at budget b." That is **literally true at b10/b25/b50 and not literally true
at b100.** Verified by set comparison, not inspection:

```bash
.venv/bin/python -c "
import lmdb, pickle, os, sys; sys.path.insert(0,'.')
from dataset import CodeRow
from train_classifier import read_manifest, build_train_items
for b,store,man in [(10,'lmdb/b10','results/manifest_tight_b10.csv'),
                    (25,'lmdb/b25','results/manifest_tight_b25.csv'),
                    (50,'lmdb/b50','results/manifest_tight_b50.csv'),
                    (100,'lmdb/train_pool_tight','results/manifest_tight.csv')]:
    env=lmdb.open(store, readonly=True, lock=False)
    with env.begin() as t:
        n=int(t.get(b'length')); gen={pickle.loads(t.get(str(i).encode())).filename for i in range(n)}
    clf={os.path.basename(p) for p,_ in build_train_items(read_manifest(man),1.0,0)}
    print(f'b{b:3d}: generator={len(gen)} classifier={len(clf)} IDENTICAL={gen==clf} clf_subset_of_gen={clf<=gen}')
"
```

**Actual:**

```
b 10: generator=215  classifier=215  IDENTICAL=True   clf_subset_of_gen=True
b 25: generator=537  classifier=537  IDENTICAL=True   clf_subset_of_gen=True
b 50: generator=1075 classifier=1075 IDENTICAL=True   clf_subset_of_gen=True
b100: generator=4274 classifier=2149 IDENTICAL=False  clf_subset_of_gen=True
```

At b100 the generator saw 2125 crops the classifier did not. **Assessed and found harmless:**
those 2125 are additional bounding-box windows cut from the *same source images*, not
additional images. Verified — the per-bbox train set and the matched train set span the
**identical 2149 base crops**:

```bash
python3 -c "
import csv
mt={r['base_crop'] for r in csv.DictReader(open('results/manifest_tight.csv')) if r['split']=='train'}
pb={r['base_crop'] for r in csv.DictReader(open('results/manifest_tight_perbbox.csv')) if r['split']=='train'}
print('matched base crops:', len(mt), 'perbbox base crops:', len(pb), 'extra bases:', len(pb-mt))
"
# actual: matched base crops: 2149  perbbox base crops: 2149  extra bases: 0
```

So: **no extra source image, no extra board, zero test-board exposure.** The b100 generator
holds extra *annotations* within images the b100 classifier already holds in full (at 100%
budget the classifier holds the entire train pool by definition). This is a wording
imprecision in the docstring, not a leak, and it cannot inflate B or C — but it is recorded
here because an audit that only checked boards would have missed it.

### 3.5 `condtight3` supplies no published number — confirmed

```bash
grep -rn "condtight" --include="*.py" --include="*.sh" . | grep -v .venv
```

**Actual — every `condtight3` consumer:**

| file | role | feeds a claimed number? |
|---|---|---|
| `run_converge.sh:13,18` | trains it | no |
| `run_cond_checks.sh:15-24` | conditioning check → `results/cond_checks.jsonl` | no |
| `run_cond_check_n360.sh:28-29` | n=360 recheck → `results/cond_check_ep320_n360*.json` | no |
| `cond_check.py:116` | default `--tag` value only | no |
| `status_checks.py:21`, `plot_consistency.py:44,112`, `build_review.py:263` | status/plot/HTML review | no |

**`condtight3` appears nowhere in `abc_budget.py`, `abc_recon.py`, `abc_b10_seeds.py`,
`sample_pool.py`, `plot_budget_abc.py`, or any `run_budget_*.sh` sampling step.** The four
synthetic pools were sampled from `prior_b{10,25,50,100}_{top,bottom}_320.pt`
(`run_budget_pipeline.sh:121-125`, `run_budget_pipeline_resume.sh:95-99`), all `warm=None`
(§3.3). The leaky numbers use no prior at all — they are VQ-VAE reconstructions
(`abc_recon.py:60-87`). `condtight3` served only as the launch gate
(`run_budget_pipeline.sh:34-46`), and the repo says so explicitly at
`run_budget_pipeline.sh:8-15` — which this audit confirms rather than takes on trust.

Cross-check that the gate's leaked status cannot help: a leaked warm-start can only *improve*
conditioning, so a gate PASS on `condtight3` is the friendliest possible test and no published
downstream number inherits its weights.

### 3.6 The leaky arm also contains no test board

The four "leaky" numbers deliberately leak *budget* (full train pool at a 10% budget) but must
not leak *boards*. Verified on the actual image files:

```bash
python3 -c "
import csv, os, collections, sys; sys.path.insert(0,'.')
from pcb_utils import parse_filename
tr={os.path.basename(r['crop_path']) for r in csv.DictReader(open('results/manifest_tight.csv')) if r['split']=='train'}
have=set(os.listdir('recon_pool'))
c=collections.Counter(parse_filename(n)['board'] for n in have)
print('recon files:', len(have), 'exact match to manifest_tight train:', tr==have)
print('boards:', dict(sorted(c.items())), 'test-board files:', {k:v for k,v in c.items() if k in ('06','09')})
"
```

**Actual:** `recon files: 2149`, `exact match to manifest_tight train: True`,
boards `{01:250, 04:425, 05:237, 07:255, 08:281, 10:143, 11:285, 12:273}`,
**test-board files: `{}` (zero)**. Consistent with `B_pool.n_train` = `n_real` + 2149 in §1.2.

**Claim 3 verdict: PASS.** Boards 06/09 appear in no classifier training set, no VQ-VAE
training set, no prior training set, and no reconstruction pool, for any of the sixteen
published numbers — directly, or indirectly via warm start (there is no warm start).

---

## Claim 4 — Non-empty synthetic pools, with actual sample counts

Context for why this claim exists: `abc_budget.py:77-82` records that an earlier run executed
with an **empty** pool and produced B and C values that were "simply A retrained —
plausible-looking, entirely meaningless." The guard is now
`assert pool` plus `assert len(pool) >= 60`. An assertion only proves what ran, so the pools
were counted on disk.

### 4.1 Actual file counts

```bash
for d in synth_b10 synth_b25 synth_b50 synth_b100; do
  echo -n "$d: files=$(ls $d/*.png | wc -l) "
  echo -n "unique_md5=$(md5sum $d/*.png | awk '{print $1}' | sort -u | wc -l) "
  echo -n "min_bytes=$(stat -c%s $d/*.png | sort -n | head -1) "
  echo "per_class=$(ls $d | sed 's/_[0-9]*\.png$//' | sort | uniq -c | tr '\n' ' ')"
done
```

**Actual:**

| pool | files on disk | distinct md5 | per class | byte range | matches `n_synth` in JSON |
|---|---|---|---|---|---|
| `synth_b10` | **360** | **360** | 60 × 6 classes | 69 444 – 96 034 | 360 ✓ |
| `synth_b25` | **360** | **360** | 60 × 6 classes | 65 585 – 98 688 | 360 ✓ |
| `synth_b50` | **360** | **360** | 60 × 6 classes | 61 652 – 98 477 | 360 ✓ |
| `synth_b100` | **360** | **360** | 60 × 6 classes | 52 140 – 92 590 | 360 ✓ |

All 1440 files are distinct (no duplicated or blank-image padding), and each is a
50–99 KB PNG — no zero-byte or degenerate files. Per-class counts are exactly 60 for
`missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper` in every pool,
matching `sample_pool.py:46` (`torch.arange(6).repeat_interleave(60)`).

### 4.2 The counts in the JSONs are internally consistent with the training-set sizes

A pool could be non-empty on disk today and still not have been used. The recorded training-set
sizes make that falsifiable: if `n_synth` were fictitious, `B.n_train` would not equal
`n_real + n_synth`.

```bash
python3 -c "
import json
for b in [10,25,50,100]:
    for r in json.load(open(f'results/abc_budget_b{b}.json')):
        print(b, r['seed'], r['n_real'], r['n_synth'], r['n_kept'],
              r['B']['n_train'], r['C']['n_train'],
              r['B']['n_train']==r['n_real']+r['n_synth'],
              r['C']['n_train']==r['n_real']+r['n_kept'],
              abs(r['keep_rate']-r['n_kept']/r['n_synth'])<1e-12)
"
```

**Actual — all 12 runs:**

| budget | seed | n_real | **n_synth** | n_kept | B n_train | C n_train | B = real+synth | C = real+kept | keep_rate ok |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 0 | 215 | **360** | 72 | 575 | 287 | ✓ | ✓ | ✓ |
| 10 | 1 | 215 | **360** | 80 | 575 | 295 | ✓ | ✓ | ✓ |
| 10 | 2 | 215 | **360** | 96 | 575 | 311 | ✓ | ✓ | ✓ |
| 25 | 0 | 537 | **360** | 195 | 897 | 732 | ✓ | ✓ | ✓ |
| 25 | 1 | 537 | **360** | 200 | 897 | 737 | ✓ | ✓ | ✓ |
| 25 | 2 | 537 | **360** | 194 | 897 | 731 | ✓ | ✓ | ✓ |
| 50 | 0 | 1075 | **360** | 184 | 1435 | 1259 | ✓ | ✓ | ✓ |
| 50 | 1 | 1075 | **360** | 175 | 1435 | 1250 | ✓ | ✓ | ✓ |
| 50 | 2 | 1075 | **360** | 177 | 1435 | 1252 | ✓ | ✓ | ✓ |
| 100 | 0 | 2149 | **360** | 114 | 2509 | 2263 | ✓ | ✓ | ✓ |
| 100 | 1 | 2149 | **360** | 113 | 2509 | 2262 | ✓ | ✓ | ✓ |
| 100 | 2 | 2149 | **360** | 102 | 2509 | 2251 | ✓ | ✓ | ✓ |

**12/12 consistent.** Every B run trained on strictly more examples than its A run, by exactly
360. The `n_synth=0` failure mode is excluded three ways: files on disk, `n_train` arithmetic,
and the `assert` at `abc_budget.py:80-82`.

For the 10-seed run, `results/abc_budget_b10_seeds.json` records `n_synth = 360` on all 10
records, and `logs/b10_seeds.log:2` prints
`budget 10%  real 215 (FROZEN subsample)  synthetic 360  test 518`.

### 4.3 Which directory each published run actually read

`abc_budget.py` does not record the `--synth` path in its JSON output. Recovered from the run
log, which prints it (`abc_budget.py:86`):

```bash
grep -n -E "^\s*(manifest|synth) |^budget [0-9]+%" logs/budget_pipeline.log
```

**Actual:**

| log line | budget | manifest | synth dir |
|---|---|---|---|
| 1274-1276 | 10% | `results/manifest_tight_b10.csv` | `synth_b10` |
| 2112-2114 | 100% | `results/manifest_tight.csv` | `synth_b100` |
| 3455-3457 | 25% | `results/manifest_tight_b25.csv` | `synth_b25` |
| 4731-4733 | 50% | `results/manifest_tight_b50.csv` | `synth_b50` |

Each budget read its own pool; no cross-wiring. Per-seed values printed in the log
(e.g. `logs/budget_pipeline.log:1283-1288`: `seed 0 | A 0.405 | B 0.467 | C 0.450`) match the
JSON to 3 dp, confirming the JSON was not edited after the run.

**CAVEAT — source-file mtime overlap for `synth_b10` only.** `sample_pool.py`
(mtime 2026-07-13 09:42:43) and `cond_check.py` (09:42:31) were modified *during* the b10
sampling window (the run started ≈08:23 and its last file landed 09:51:51; a concurrent
`cond_check` process recompiled `__pycache__/cond_check.cpython-311.pyc` at 09:43:16, matching
the n=360 recheck described at `run_budget_pipeline_resume.sh:4-6`). A mid-run edit cannot
affect an already-loaded Python process, and the mitigating evidence is strong: the b10 log's
output format (`  36/360 (528s)` … `wrote 360 labelled synthetic crops -> synth_b10/  (5305s)`)
matches the current `sample_pool.py:59-60` format strings exactly, and all four pools share an
identical structure (10 chunks × 36, 60 per class). But strictly: for `synth_b10` alone the
on-disk source is not *provably* byte-identical to the source that produced it. `synth_b25`,
`synth_b50` and `synth_b100` were all sampled after the edit and are unaffected. This does not
change the verdict — the pool exists, is non-empty, and contains 360 distinct crops — but it is
recorded rather than glossed.

**Claim 4 verdict: PASS.** Every A/B/C run at every budget used a real, non-empty, 360-crop,
60-per-class synthetic pool of distinct images.

---

## Claim 5 — The filter at budget b was the Condition-A classifier at budget b

**Source:** `abc_budget.py:89-99` (the 3-seed runs) and `abc_b10_seeds.py:69-70` (the 10-seed
run). `train_clf` and `filter_pool` are imported from `abc_recon.py:37` / `abc_b10_seeds.py:32`.

**Exact mechanism, read from source:**

```python
# abc_budget.py:91-98
for s in SEEDS:
    # --- A: real only. This model IS the filter (no-leakage rule). ---
    fmodel, f1a, acca, pera = train_clf(real, vl, device, s, args.epochs)
    kept = filter_pool(fmodel, pool, device)

    rec = {..., 'A': {'macro_f1': f1a, ...}}
    for name, items in (('B', real + pool), ('C', real + kept)):
```

This is the strongest form the claim can take: **`fmodel` is the same Python object** that
produced the reported `A` macro-F1 (`f1a`) and that filtered the pool. There is no second
training call, no checkpoint reload, no separate filter model, and no opportunity for the two
to diverge. Verified point by point:

| requirement | evidence | verdict |
|---|---|---|
| Filter is the Condition-A model, not a separate model | single `train_clf(...)` call; `fmodel` returned and passed straight to `filter_pool` (`abc_budget.py:92-93`) | PASS |
| Same **budget** | `real` is the only training input, built from that budget's manifest (`abc_budget.py:62-73`) | PASS |
| Same **seed** | both inside `for s in SEEDS`; `train_clf(..., s, ...)` calls `set_seed(s)` (`abc_recon.py:91`) | PASS |
| Filter sees **no** test data | `filter_pool` reads only `pool` paths (`abc_recon.py:110-119`) | PASS |
| Filter rule = keep iff prediction equals the conditioned label | `kept += [it for it, p in zip(chunk, pred) if p == it[1]]` (`abc_recon.py:118`); `it[1]` is `CLASS_TO_IDX[cls]` parsed from the pool filename, i.e. the class the prior was conditioned on (`abc_budget.py:42-49`, `sample_pool.py:55-56`) | PASS |
| No confidence threshold / no tuning | `argmax(1)` only, no threshold parameter anywhere in `filter_pool` | PASS |
| Same rule in the 10-seed run | `abc_b10_seeds.py:69-70`, identical two-line pattern | PASS |

**Command to re-read the exact lines:**

```bash
sed -n '89,99p' abc_budget.py          # A trained, then handed to the filter
sed -n '109,119p' abc_recon.py         # filter_pool: keep iff pred == conditioned label
sed -n '42,49p' abc_budget.py          # label parsed from pool filename
sed -n '66,75p' abc_b10_seeds.py       # same rule, 10-seed run
grep -n "filter = the budget" logs/budget_pipeline.log
```

**Runtime confirmation** (the pipeline echoes the rule at each budget, and the resulting keep
rates vary by seed exactly as a per-seed model must):

```
logs/budget_pipeline.log:1273  [10%]  6/6 A/B/C  (filter = the budget-10 Condition-A model, SAME seed)
logs/budget_pipeline.log:2111  [100%] 6/6 A/B/C  (filter = the budget-100 Condition-A model, SAME seed)
logs/budget_pipeline.log:3454  [25%]  6/6 A/B/C  (filter = the budget-25 Condition-A model, SAME seed)
logs/budget_pipeline.log:4730  [50%]  6/6 A/B/C  (filter = the budget-50 Condition-A model, SAME seed)
```

Corroborating signature in the data: `n_kept` differs across seeds *within* a budget
(b10: 72 / 80 / 96; b100: 114 / 113 / 102). A shared or budget-mismatched filter would produce
identical `n_kept` across seeds. It does not.

Keep rates by budget (from `keep_rate`, mean over 3 seeds): 10% → 23.0%, 25% → 55.3%,
50% → 49.6%, 100% → 30.5%.

**Claim 5 verdict: PASS.**

---

## Provenance of the audited artefacts

Checked so that "the script on disk is the script that ran" is not assumed:

```bash
git status --porcelain abc_budget.py abc_recon.py abc_b10_seeds.py train_classifier.py \
                       sample_pool.py pcb_utils.py make_budget_subsamples.py build_split.py
stat -c "%y  %n" abc_budget.py abc_recon.py abc_b10_seeds.py results/abc_budget_b*.json results/abc_recon.json
```

- All audited scripts are **tracked and unmodified** relative to `HEAD` (empty `git status`).
- Every script predates the result file it produced:
  `abc_recon.py` 07-12 15:10 → `abc_recon.json` 07-12 15:41;
  `abc_budget.py` 07-13 07:22 → `abc_budget_b10.json` 07-13 09:53, `b100` 07-14 05:08,
  `b25` 07-14 08:45, `b50` 07-14 14:11;
  `abc_b10_seeds.py` 07-13 23:16 → `abc_budget_b10_seeds.json` 07-13 23:22.
- Exception, already recorded: `sample_pool.py` / `cond_check.py` vs `synth_b10` (§4.2).

## Two further observations an adversarial reader should have

1. **Runs are not bit-reproducible.** At b10 seed 0 the Condition-A macro-F1 is 0.405454 in
   `abc_budget_b10.json` and 0.409410 in `abc_budget_b10_seeds.json` — same data, same seed,
   same recipe, different run. `set_seed` (`train_classifier.py:26-30`) does not set
   `torch.backends.cudnn.deterministic`, and the loaders use `num_workers=8`. Re-running will
   reproduce these numbers to roughly ±0.004, not exactly. Every claim above was verified
   against the recorded values, which is the right standard for an audit of *published*
   numbers, but a reviewer asking for exact re-execution will not get bitwise agreement.
2. **Error bars are narrower than a full study's.** The data subsample is frozen per budget, so
   the ±values are over classifier initialisation only, not data resampling
   (`abc_budget.py:18-21`, `make_budget_subsamples.py:157-159`). The repo states this; it is
   true; it limits generalisation over draws of the training subsample. This is disclosed, not
   concealed — but it means the 3-seed ± figures should never be read as sampling error.

---

## Verdict

| # | Claim | Verdict |
|---|---|---|
| 1 | Twelve A/B/C numbers + four leaky numbers, from raw JSONs | **PASS** — 16/16 exact |
| 2 | 10-seed paired test recomputed from per-seed values | **PASS** — 4/4 exact |
| 3 | No test-board leakage, incl. warm-start lineage; `condtight3` unused | **PASS** — 8/8 priors `warm=None`; 0 test-board rows in every training store; caveat §3.4 |
| 4 | Non-empty synthetic pools with actual counts | **PASS** — 360 distinct crops × 4 budgets, 60/class; caveat §4.2 |
| 5 | Filter at budget b = Condition-A classifier at budget b | **PASS** — same object, same seed |

Nothing in the five audited claims is void. The two caveats (§3.4 b100 generator corpus is a
per-bbox superset over the same 2149 base images; §4.2 `synth_b10` source-file mtime overlap)
are disclosed in full and neither changes a published value.

---

# §6 — Every remaining ICETA number

Added 2026-07-28. Same rules as §1–§5: verified from raw result JSONs, raw CSVs, the VOC
annotation XML and source code only. Summary documents (`findings.md`, `results/paper_tables.md`)
were used **only** to locate which raw field a claim refers to, never as evidence — and where
they disagree with the raw files, that is recorded below as a finding.

**26 claims checked: 20 PASS, 4 MISMATCH, 2 PASS-with-qualification.**

A number is marked **[derived]** when it is not a recorded field but a restatement computed
from other numbers (a difference, ratio or share). Derived numbers cannot be "wrong" in the
file — they can only be wrong in the arithmetic or in the choice of inputs, and for two of
them the input choice is the problem.

## 6.1 Summary

| # | claim | verdict |
|---|---|---|
| 1 | 600px baseline 0.246 ± 0.013 | **PASS** |
| 2 | TIGHT size-matched 0.900 ± 0.018 | **MISMATCH** — actual 0.897 ± 0.020 |
| 3 | 93% of the gain attributable to crop scale [derived] | **PASS** |
| 4 | Binary baseline 0.838 @ 10% | **MISMATCH** — that is seed 0; 3-seed mean is 0.841 |
| 5 | Binary honest +0.018 @ 10% [derived] | **PASS** |
| 6 | Binary seed noise ±0.021 | **PASS** |
| 7 | Binary ≤ +0.007 at other budgets [derived] | **PASS on rounding** — true max +0.0074 |
| 8 | Binary filter rejection 64–95% [derived] | **PASS** |
| 9 | Conditioning consistency ≤ 53% at any budget | **PASS** |
| 10 | Conditioning 15% at 215 crops | **PASS** |
| 11 | Filter keep-rate 25.5% at b10 | **PASS** — 10-seed run only; 3-seed run is 23.0% |
| 12 | Reconstructions classify at 0.918 | **PASS** |
| 13 | Held-out prior NLL worse than a uniform prior | **MISMATCH if unqualified** — true of the bottom prior, false of the top |
| 14 | Harm −0.046 / −0.044 / −0.052 [derived] | **PASS** |
| 15 | Leaky gain +0.49 [derived] | **MISMATCH** — cross-experiment subtraction; within-experiment is +0.51 |
| 16 | 18× the detection floor [derived] | **PASS on rounding** — pairing- and formula-dependent (15.7×–18.3×) |
| 17 | MDE ±0.028 [derived] | **PASS** under the normal approximation; exact paired-t gives ±0.031 |
| 18 | Full-data baseline 0.918 | **PASS but ambiguous** — two distinct quantities both round to 0.918 |
| 19 | Median defect 27×27 px | **PASS** |
| 20 | ≈0.2% of frame [derived] | **PASS** (area fraction) |
| 21 | 518 test crops | **PASS** |
| 22 | 2 149 primary train-pool crops | **PASS** |
| 23 | Budgets 215 / 537 / 1 075 / 2 149 | **PASS** |
| 24 | 10 boards | **PASS** |
| 25 | 6 classes | **PASS** |
| 26 | Primary = defect nearest the frame centre | **PASS** |

---

## 6.2 Crop scale

**Source:** `results/classifier_A_b100_s{0,1,2}.json` (600px), `classifier_Atight_b100_s*.json`
(TIGHT size-matched), `classifier_Atightpb_b100_s*.json` (TIGHT per-bbox).
**Field:** `macro_f1`, mean and population std over the 3 seed files.

```bash
python3 -c "
import json, glob, statistics as st
for cond in ['A','Atight','Atightpb']:
    fs=sorted(glob.glob(f'results/classifier_{cond}_b100_s*.json'))
    v=[json.load(open(f))['macro_f1'] for f in fs]
    n=json.load(open(fs[0]))['n_train']
    print(f'{cond:9s} n_train={n:5d}  {st.mean(v):.6f} +/- {st.pstdev(v):.6f}  per-seed {[round(x,6) for x in v]}')
"
```

| set | n_train | claimed | **actual** | verdict |
|---|---|---|---|---|
| 600px baseline | 2149 | 0.246 ± 0.013 | **0.246396 ± 0.012840** | **PASS** |
| TIGHT size-matched | 2149 | 0.900 ± 0.018 | **0.897488 ± 0.020431** | **MISMATCH** |
| TIGHT per-bbox | 4274 | (not in scope) | 0.946600 ± 0.004534 | — |

### Claim 2 is a MISMATCH, and there are three different values in circulation

Per-seed TIGHT size-matched `macro_f1`: **0.925527, 0.889510, 0.877426**. Mean 0.897488,
population std 0.020431, sample std 0.025023.

No field in those three files yields 0.900 ± 0.018. Every candidate:

| field | mean ± pop-std | ± sample-std |
|---|---|---|
| `macro_f1` | 0.897488 ± 0.020431 | ± 0.025023 |
| `accuracy` | 0.895753 ± 0.021206 | ± 0.025972 |
| `best_macro_f1` | 0.918944 ± 0.004669 | ± 0.005718 |

Three values are in circulation for this one cell, and they do not agree:

- the paper: **0.900 ± 0.018** — matches no field;
- `findings.md:34` and `:60`: **0.896 ± 0.018** — the mean matches `accuracy` (0.895753),
  not macro-F1, and no field has std 0.018;
- `results/paper_tables.md:8`: **0.897 ± 0.020** — **correct**, matches `macro_f1` exactly.

The generated table (`paper_tables.md`) is right; the two hand-written restatements drifted
from it in different directions. The only 0.018 anywhere nearby is the *600px baseline's*
`best_macro_f1` population std (0.018084), which is a different row and a different field —
noted as the likeliest transcription origin, not as an established one.

**Use 0.897 ± 0.020.** Nothing else in the paper depends on which of the three is printed —
see claim 3.

#### Out-of-scope correction made 2026-07-30: the accuracy cell in the same row

This claim was scoped to macro-F1, so the *accuracy* column of `findings.md:34` was not
checked when it was written. It is also wrong. The row read:

```
| TIGHT matched | 2149 | 0.896 ± 0.018 | 0.898 |
                          ^ macro-F1      ^ accuracy
```

Recomputed from the same three files, `accuracy` is **0.895753 ± 0.021206** (population std).
No field yields 0.898. Both cells were corrected together, to `0.897 ± 0.020` and `0.896`,
because leaving a demonstrably wrong number in the row while fixing its neighbour would have
left a public reader with the same problem this audit exists to remove — three values in
circulation for one result.

`results/paper_tables.md:8` already printed `0.896 ± 0.021` for accuracy and needed no change.
After the correction `findings.md`, `results/paper_tables.md` and this file agree on both
fields.

The correction is recorded here rather than folded into claim 2 above, because claim 2's
verdict was reached on a narrower question and its scope should not be rewritten after the
fact. Nothing derived depends on the accuracy figure.

### Claim 3 — the 93% decomposition **[derived]**

```bash
python3 -c "
import json, glob, statistics as st
m=lambda c: st.mean([json.load(open(f))['macro_f1'] for f in sorted(glob.glob(f'results/classifier_{c}_b100_s*.json'))])
scale=m('Atight')-m('A'); size=m('Atightpb')-m('Atight')
print('scale %+.6f  size %+.6f  total %.6f  scale share %.2f%%' % (scale,size,scale+size,100*scale/(scale+size)))
"
```

**Actual:** crop scale `0.897488 − 0.246396 = +0.651092`; dataset size
`0.946600 − 0.897488 = +0.049112`; total `0.700204`; **scale share = 92.99%**, size share 7.01%.

**PASS.** Rounds to the published "+0.65 from scale, +0.05 from size, ~93% / ~7%". The
decomposition is robust to claim 2's discrepancy: substituting the claimed 0.900 gives a
93.3% share, still 93%. The two effects are measured at held-fixed dataset size (both 2149)
and held-fixed crop scale respectively, so the decomposition is not confounded.

---

## 6.3 Binary track

**Source:** `results/abc_binary.json` — 12 records (4 budgets × 3 seeds), keys `A`,
`B_honest`, `C_honest`, `B_leaky`, `C_leaky`. **Field:** `[i][cond]["macro_f1"]`, and
`[i]["C_*"]["keep_rate"]`.

```bash
python3 -c "
import json, statistics as st
d=json.load(open('results/abc_binary.json'))
for b in [0.1,0.25,0.5,1.0]:
    r=[x for x in d if abs(x['budget']-b)<1e-9]
    a=st.mean([x['A']['macro_f1'] for x in r])
    print('%3d%% A=%.6f +/- %.6f  n_real=%d' % (int(b*100), a, st.pstdev([x['A']['macro_f1'] for x in r]), r[0]['n_real']))
    for c in ['B_honest','C_honest','B_leaky','C_leaky']:
        v=[x[c]['macro_f1'] for x in r]
        print('      %-9s %.6f +/- %.6f  (%+.6f vs A)' % (c, st.mean(v), st.pstdev(v), st.mean(v)-a))
"
```

### Claim 4 — baseline 0.838 @ 10% is a **MISMATCH**

| quantity | value |
|---|---|
| per-seed A @ 10% | 0.837838, 0.848900, 0.836357 |
| **3-seed mean** | **0.841031 ± 0.005607** |
| seed 0 alone | **0.837838** → 0.838 |

The published 0.838 is **seed 0 in isolation**, not the 3-seed mean that every other number in
this table uses. `findings.md:155` reports the same cell correctly as 0.841 ± 0.006. The
correct baseline is **0.841**.

This matters for claim 5: quoting the baseline as 0.838 while quoting the gain as +0.018
(which is computed against 0.841) is internally inconsistent — 0.859 − 0.838 would be +0.021,
not +0.018.

### Claims 5–8

| # | claim | source | **actual** | verdict |
|---|---|---|---|---|
| 5 | honest +0.018 @ 10% **[derived]** | `B_honest` − `A` | 0.858913 − 0.841031 = **+0.017882** | **PASS** |
| 6 | seed noise ±0.021 | `B_honest` @ 10% pop-std | **0.021254** | **PASS** |
| 7 | ≤ +0.007 at other budgets **[derived]** | max honest gain, 25/50/100% | **+0.007412** (25%, `B_honest`) | **PASS on rounding** |
| 8 | filter rejection 64–95% **[derived]** | `1 − C_leaky.keep_rate`, all 12 runs | **64.4% – 94.7%** | **PASS** |

Claim 7 detail — every honest gain outside the 10% budget:

| budget | `B_honest` − A | `C_honest` − A |
|---|---|---|
| 25% | **+0.007412** | −0.014199 |
| 50% | +0.001618 | +0.000005 |
| 100% | +0.001287 | −0.004827 |

The maximum is +0.007412, which **rounds** to +0.007 but is not ≤ 0.007 exactly. `findings.md`
states the same bound more defensibly as "< +0.008 everywhere else". Prefer that phrasing, or
write "≤ +0.008". Note also that `C_honest` is *negative* at 25% and 100%, so "≤ +0.007" is a
bound on the best case, not a description of typical behaviour.

Claim 8 detail — `C_leaky` rejection rates, all 12 runs: 94.7, 78.9, 64.7 (10%); 76.9, 68.1,
64.4 (25%); 78.1, 70.3, 88.9 (50%); 75.6, 73.3, 79.4 (100%). Min 64.4%, max 94.7% → **64–95%**.
The corresponding honest keep-rates span 20.6%–91.4%, consistent with `findings.md`'s "21–91%".

---

## 6.4 Conditioning consistency

**Source:** `results/cond_checks_budget.jsonl` — one record per budget, tag `prior_b{10,25,50,100}`.
**Field:** `consistency`. All four are **n = 72** checks (12 per class × 6 classes).

```bash
python3 -c "
import json
for ln in open('results/cond_checks_budget.jsonl'):
    r=json.loads(ln); print('%-12s n=%d consistency=%.4f (%.1f%%) V=%.4f collapse=%.4f' %
        (r['tag'], r['n'], r['consistency'], 100*r['consistency'], r['cramers_v'], r['collapse_index']))
"
```

| tag | n | **consistency** | Cramér's V | collapse index |
|---|---|---|---|---|
| `prior_b10` | 72 | **15.28%** | 0.2371 | 0.4028 |
| `prior_b25` | 72 | **50.00%** | 0.5015 | 0.4028 |
| `prior_b50` | 72 | **52.78%** | 0.5725 | 0.4583 |
| `prior_b100` | 72 | **37.50%** | 0.4732 | 0.4583 |

| # | claim | **actual** | verdict |
|---|---|---|---|
| 9 | ≤ 53% at any budget | max = **52.78%** (b50) | **PASS** |
| 10 | 15% at 215 crops | b10 = **15.28%**, and `abc_budget_b10.json` records `n_real = 215` | **PASS** |

Both PASS. Two things the paper should carry alongside them: these are n = 72 checks, so the
standard error on a ~50% estimate is ≈5.9 pp — the b25/b50 difference (50.0 vs 52.8) is well
inside noise and must not be read as a trend. And **every budget has `collapsed: true`**
(collapse index 0.40–0.46, all ≥ the 0.40 pre-registered line), which is the finding that
makes the consistency numbers interpretable at all.

---

## 6.5 Filter keep-rate at b10

**Source:** `results/abc_budget_b10_seeds.json`, field `keep_rate` (= `n_kept / n_synth`).

```bash
python3 -c "
import json, statistics as st
d=json.load(open('results/abc_budget_b10_seeds.json'))
print('10-seed mean keep_rate = %.6f -> %.1f%%' % (st.mean([r['keep_rate'] for r in d]), 100*st.mean([r['keep_rate'] for r in d])))
print('3-seed  mean keep_rate = %.1f%%' % (100*st.mean([r['keep_rate'] for r in json.load(open('results/abc_budget_b10.json'))])))
"
```

**Actual:** per-seed 20.6, 23.1, 27.2, 25.8, 25.3, 26.9, 25.3, 23.3, 29.7, 27.8 %;
mean = **0.255000 exactly → 25.5%** (n_kept 74+83+98+93+91+97+91+84+107+100 = 918; 918/3600).

**Claim 11: PASS**, with a qualification the paper must state: **25.5% is the 10-seed run.**
The 3-seed `abc_budget_b10.json` — the run behind Table 1's b10 row — gives **23.0%**. Both
are correct for their own run; quoting 25.5% next to Table 1's b10 column mixes them.

---

## 6.6 Reconstructions and prior NLL

### Claim 12 — reconstructions classify at 0.918: **PASS**

**Source:** `results/recon_sanity.json`, field `macro_f1`.
**Actual: 0.918217** (n = 518, judge `checkpoint/classifier_Atight_b100_s0.pt`, VQ-VAE
`checkpoint/vqvae_tight_560.pt`).

```bash
python3 -c "import json; d=json.load(open('results/recon_sanity.json')); print(d['macro_f1'], d['n'], d['judge'], d['vqvae'])"
```

The paired framing in `findings.md:80` — "reads real crops at 0.926 and VQ-VAE reconstructions
at 0.918" — is internally consistent: the same seed-0 judge scores **0.925527** on the real
518 test crops (`classifier_Atight_b100_s0.json`) and **0.918217** on reconstructions of those
same crops. Same model, same images, one round-trip apart. Both are single-seed values, which
is correct here — it is a paired instrument check, not a 3-seed result.

### Claim 13 — "held-out prior NLL is worse than a uniform prior": **MISMATCH if unqualified**

**Source:** `results/prior_nll.json`, field `[i]["top"|"bottom"]["heldout_nll"]`.
**Reference:** a uniform prior over the 512-way codebook has NLL = ln(512) = **6.238 nats/code**
(`eval_prior_nll.py:37` `CHANCE = math.log(512)`, documented at line 21).

```bash
python3 -c "
import json, math
ch=math.log(512); print('uniform =', round(ch,4), 'nats/code')
for r in json.load(open('results/prior_nll.json')):
    print('ep %3d  top %.4f (%s)   bottom %.4f (%s)' % (r['epoch'],
        r['top']['heldout_nll'], 'WORSE' if r['top']['heldout_nll']>ch else 'better',
        r['bottom']['heldout_nll'], 'WORSE' if r['bottom']['heldout_nll']>ch else 'better'))
"
```

| epoch | top held-out NLL | vs uniform | bottom held-out NLL | vs uniform |
|---|---|---|---|---|
| 80 | 3.4173 | better | 7.6501 | **worse** |
| 120 | 3.9681 | better | 8.5653 | **worse** |
| 160 | 4.2677 | better | 9.0997 | **worse** |
| 200 | 4.6529 | better | 9.6714 | **worse** |
| 240 | 4.7885 | better | 10.0645 | **worse** |
| 280 | 5.0465 | better | 10.4924 | **worse** |
| 320 | 5.1235 | better | 10.8456 | **worse** |

**The claim is true of the bottom prior at every recorded epoch, and false of the top prior at
every recorded epoch** (5.12 < 6.238 at epoch 320 — the top prior beats uniform by 1.1 nats).

As written the claim overstates. The defensible statement is: *"the **bottom** prior's held-out
NLL is worse than a uniform prior over the codebook at every epoch (10.85 vs 6.24 nats/code at
320), and rises monotonically with training; the top prior stays below uniform (5.12) but also
rises monotonically."* Both hierarchies show held-out NLL **increasing** while train NLL falls
(top 0.0092, bottom 0.1960 at epoch 320) — that is the overfitting result, and it holds for
both.

One nuance worth a clause: the bottom prior's held-out *accuracy* is 4.78%, far above the
1/512 = 0.195% a uniform prior gives. So it is not that the bottom prior knows nothing — it is
badly calibrated, confidently wrong often enough to lose to uniform on log-loss.

---

## 6.7 Harm figures — claim 14 **[derived]**

**Source:** `results/abc_budget_b{25,50,100}.json`. **Field:** mean `B.macro_f1` − mean `A.macro_f1`.

```bash
python3 -c "
import json, statistics as st
for b in [25,50,100]:
    d=json.load(open(f'results/abc_budget_b{b}.json'))
    a=st.mean([r['A']['macro_f1'] for r in d]); x=st.mean([r['B']['macro_f1'] for r in d])
    print('b%-4d A=%.6f B=%.6f  B-A=%+.6f' % (b,a,x,x-a))
"
```

| budget | claimed | **actual** | verdict |
|---|---|---|---|
| 25% | −0.046 | **−0.045687** | **PASS** |
| 50% | −0.044 | **−0.044213** | **PASS** |
| 100% | −0.052 | **−0.051530** | **PASS** |

All three **PASS**. (At 10% the difference is +0.012, positive — which is why the claim is
scoped to "above 10%".) These are differences of 3-seed means, not paired per-seed statistics;
the only budget with a paired test is b10 (§2), where the effect is null.

---

## 6.8 Derived claims

### Claim 15 — "+0.49 leaky gain": **MISMATCH (cross-experiment subtraction)**

The leaky ceiling `B_pool` at b10 is **0.909555** (`abc_recon.json`, verified §1.2). The gain
depends entirely on *which* Condition-A it is subtracted from — and there are three, from three
different runs:

```bash
python3 -c "
import json, statistics as st
R=json.load(open('results/abc_recon.json'))
bp=st.mean([r['B_pool']['macro_f1'] for r in R if abs(r['budget']-0.1)<1e-9])
ar=st.mean([r['A']['macro_f1'] for r in R if abs(r['budget']-0.1)<1e-9])
ab=st.mean([r['A']['macro_f1'] for r in json.load(open('results/abc_budget_b10.json'))])
as_=st.mean([r['A']['macro_f1'] for r in json.load(open('results/abc_budget_b10_seeds.json'))])
print('B_pool %.6f' % bp)
for n,a in [('abc_recon A (SAME file)',ar),('abc_budget_b10 A',ab),('abc_budget_b10_seeds A',as_)]:
    print('  minus %-26s %.6f = %+.6f -> %+.2f' % (n,a,bp-a,bp-a))
"
```

| Condition-A used | value | gain | rounds to |
|---|---|---|---|
| **`abc_recon.json` — the same file, same run** | 0.399530 | **+0.510025** | **+0.51** |
| `abc_budget_b10.json` — a different experiment | 0.418598 | +0.490957 | +0.49 |
| `abc_budget_b10_seeds.json` — a third run | 0.432452 | +0.477103 | +0.48 |

**+0.49 is obtainable only by subtracting one experiment's Condition-A from another
experiment's B_pool.** `abc_recon.py` trains its own Condition-A inline, in the same process,
on the same subsample, for exactly this comparison (`abc_recon.py:143`), and that model scores
0.399530. The within-experiment gain is **+0.51**, which is what `findings.md:124` and
`plot_budget_abc.py:11` both report.

The three A values differ because of the run-to-run nondeterminism documented in §"Two further
observations" (±0.004) *plus*, for the 10-seed file, a different seed set. Mixing them is not a
rounding difference; it is a comparison between two models that never met.

**Use +0.51.** If the paper needs the more conservative figure it should say which two runs are
being differenced and why.

### Claims 16–17 — detection floor and MDE **[derived]**

```bash
python3 -c "
import json, math, statistics as st
from scipy import stats
d=json.load(open('results/abc_budget_b10_seeds.json'))
diff=[r['B']['macro_f1']-r['A']['macro_f1'] for r in d]; n=len(diff)
se=st.stdev(diff)/math.sqrt(n)
z=(stats.norm.ppf(.975)+stats.norm.ppf(.80))*se
t=(stats.t.ppf(.975,n-1)+stats.t.ppf(.80,n-1))*se
print('sd=%.6f se=%.6f' % (st.stdev(diff), se))
print('MDE normal-approx = %.6f ; MDE exact-t = %.6f' % (z,t))
for g,lbl in [(0.510025,'+0.510'),(0.490957,'+0.491')]:
    print('  %s / %.6f = %.2fx   |   / %.6f = %.2fx' % (lbl,z,g/z,t,g/t))
"
```

**Actual:** sd of paired differences 0.031473, se 0.009953.

| MDE formula | value | rounds to |
|---|---|---|
| normal approximation, (z₀.₉₇₅ + z₀.₈₀)·se | **0.027883** | **0.028** |
| exact paired-t, (t₀.₉₇₅,₉ + t₀.₈₀,₉)·se | **0.031306** | 0.031 |

**Claim 17: PASS** under the normal approximation — which is what `logs/b10_seeds.log:33-35`
used ("about +0.0279"). With only 10 seeds the exact t-based MDE (**±0.031**) is the more
honest figure; the normal approximation understates the detectable effect by ~11% at df = 9.
Either is defensible if stated; **±0.028 should be labelled as the normal-approximation value.**

**Claim 16: PASS on rounding**, but it is doubly contingent:

| gain ÷ MDE | normal-approx MDE | exact-t MDE |
|---|---|---|
| +0.510 (within-experiment) | **18.29×** | 16.29× |
| +0.491 (cross-experiment) | 17.61× | 15.68× |

Only the top-left cell gives "18×". With the same +0.51 gain but the exact-t MDE it is 16×;
with the cross-experiment +0.49 gain it is 18× only after rounding 17.61. The claim survives
as stated, but "**more than an order of magnitude above what this design could detect**" is
robust across all four cells and is what the paper should say.

### Claim 18 — full-data baseline 0.918: **PASS but ambiguous**

Two *different* quantities both round to 0.918, and the paper uses "0.918" for both:

| quantity | source | value |
|---|---|---|
| classifier on **real** full-data train pool, 3-seed mean | `abc_recon.json` A @ budget 1.0 | **0.917775** |
| seed-0 judge on **VQ-VAE reconstructions** of the 518 test crops | `results/recon_sanity.json` | **0.918217** |

Both round to 0.918 by coincidence, not by construction — they measure different things (a
training-data ceiling vs a decoder-fidelity check). The "leaky ceiling" sentence in
`plot_budget_abc.py:11-12` ("0.910 against a full-data ceiling of 0.918") means the **first**.
The "reconstructions classify at 0.918" sentence means the **second**. Both are correct;
using one number for both invites the reader to think a single measurement is being reused.
Recommend printing them to a decimal that separates them, or naming each explicitly.

---

## 6.9 Dataset facts

Computed from the VOC annotation XML — the primary source — not from any derived manifest.

```bash
python3 -c "
import glob, os, statistics as st, sys
import xml.etree.ElementTree as ET
sys.path.insert(0,'.')
from pcb_utils import parse_filename
ws=hs=None; ws=[]; hs=[]; areas=[]; n=0
for x in sorted(glob.glob('VOC_PCB/Annotations/*.xml')):
    p=parse_filename(os.path.basename(x)[:-4]+'.jpg')
    if p is None or p['variant']!='plain': continue
    n+=1
    for o in ET.parse(x).getroot().findall('object'):
        b=o.find('bndbox')
        w=int(b.find('xmax').text)-int(b.find('xmin').text)
        h=int(b.find('ymax').text)-int(b.find('ymin').text)
        ws.append(w); hs.append(h); areas.append(w*h)
print('plain base crops %d ; bboxes %d' % (n,len(ws)))
print('median w=%.1f h=%.1f ; median area=%.1f px2 = %.4f%% of 600x600' % (st.median(ws),st.median(hs),st.median(areas),100*st.median(areas)/360000))
"
```

| # | claim | source | **actual** | verdict |
|---|---|---|---|---|
| 19 | median defect 27×27 px | VOC XML, 5 416 bboxes over 2 667 plain crops | **median w = 27.0, median h = 27.0** | **PASS** |
| 20 | ≈0.2% of frame **[derived]** | median bbox area ÷ 600² | **754 px² = 0.2094%**; 27×27 = 729 px² = 0.2025% | **PASS** |
| 21 | 518 test crops | `manifest_tight.csv`, `split==test & variant==plain` | **518** (boards 06, 09) | **PASS** |
| 22 | 2 149 primary train-pool crops | `manifest_tight.csv`, `split==train & variant==plain` | **2 149** | **PASS** |
| 23 | budgets 215 / 537 / 1 075 / 2 149 | `abc_budget_b*.json` `n_real`; `budget_subsamples.json` `n_train` | **215 / 537 / 1 075 / 2 149**, identical across seeds, and agreeing across both files | **PASS** |
| 24 | 10 boards | `results/splits.json` | **8 train + 2 test = 10** | **PASS** |
| 25 | 6 classes | `results/splits.json` `classes` | **6** | **PASS** |
| 26 | primary = defect nearest the frame centre | `make_tight_crops.py:63` | `prim = argmin(hypot(cx-300, cy-300))` | **PASS** |

**Claim 20 — which "fraction of frame".** Both the paper's ≈0.2% (area) and `findings.md:51`'s
"~4.5% of frame" are arithmetically right but measure different things: 0.2% is the **area**
fraction (729/360 000), 4.5% is the **linear** fraction (27/600). For an argument about how
many pixels the defect occupies — which is the paper's argument — **0.2% (area) is the correct
one**. `findings.md`'s "~4.5% of frame" should read "~4.5% of frame width".

**Claim 26 — verified in code, and it binds.** 1 687 of 2 667 base crops (63%) contain more
than one annotated defect (distribution: 980 single, 934 two, 492 three, 213 four, 48 five), so
the primary rule is doing real work, not describing an edge case. It also explains the two
manifest sizes exactly: 5 416 total bboxes = 4 274 train (all of them, the per-bbox set) +
1 142 test; the matched set keeps one primary per base crop, giving 2 149 train + 518 test.
Those five counts are mutually consistent across `manifest_tight.csv`,
`manifest_tight_perbbox.csv`, `manifest_nodefect.csv` and the raw XML.

---

## 6.10 What to change in the paper

Ranked by consequence. Nothing here changes a conclusion; four items change a printed number.

1. **TIGHT size-matched is 0.897 ± 0.020, not 0.900 ± 0.018** (claim 2). Take the value from
   `results/paper_tables.md`, which is generated; the hand-written restatements in
   `findings.md` are also wrong, differently.
2. **The binary baseline at 10% is 0.841, not 0.838** (claim 4). 0.838 is seed 0 alone, and
   pairing it with the +0.018 gain is internally inconsistent.
3. **The leaky gain is +0.51, not +0.49** (claim 15). +0.49 subtracts a different experiment's
   Condition-A from `abc_recon`'s B_pool.
4. **Scope the NLL claim to the bottom prior** (claim 13). The top prior's held-out NLL is
   *better* than uniform at every epoch.
5. Label **±0.028 as the normal-approximation MDE** (claim 17); the exact paired-t value is
   ±0.031. Prefer "more than an order of magnitude above the detection floor" to "18×"
   (claim 16), which holds only for one of four defensible pairings.
6. State that the **25.5% keep-rate is the 10-seed run** (claim 11); Table 1's b10 row rests on
   the 3-seed run, where it is 23.0%.
7. Write the binary bound as **"< +0.008"** rather than "≤ +0.007" (claim 7); the true maximum
   is +0.0074.
8. Disambiguate the **two 0.918s** (claim 18) — full-data real baseline 0.9178 vs reconstruction
   fidelity 0.9182.
9. In `findings.md`, "~4.5% of frame" should read **"~4.5% of frame width"** (claim 20).

---

## §6 verdict

Of 26 numbers: **20 PASS**, **2 PASS with a qualification that must be printed** (claims 7, 16),
**4 MISMATCH** (claims 2, 4, 13, 15).

Every mismatch is a transcription or pairing error in the write-up, not an error in the
experiments: in all four cases the raw result files contain the correct value, and in three of
the four a generated artefact (`results/paper_tables.md`, `logs/b10_seeds.log`,
`plot_budget_abc.py`'s own summary) already prints it correctly. No re-run is required to fix
any of them. The direction and significance of every claim survives correction — the corrected
crop-scale effect is +0.651 instead of +0.654, the corrected binary gain is +0.018 against a
0.841 baseline, and the corrected leaky gain is larger, not smaller, than published.

> **Superseded in part by §7 (2026-07-28).** The last clause above accepts the paper's framing
> that `abc_recon`'s `B_pool` measures leakage. §7 shows it does not: it is a reconstruction
> upper bound, and the true leaky protocol yields **+0.002**, not +0.51. Claim 15's arithmetic
> verdict stands (+0.510 is the correct within-experiment value of *that* quantity); what
> changes is what the quantity means.

---

# §7 — The true leaky control (new experiment, 2026-07-28)

§1–§6 verified numbers against files. This section reports a **new experiment**, run because
§6.8 found that the quantity the paper calls "leaky" is not a generator experiment at all.

## 7.1 Why the published leaky number does not measure leakage

The paper's Finding 5c — *"the number the field would have reported is leakage"* — cites
`abc_recon.json`'s `B_pool`. Read from source, that condition is:

```python
# abc_recon.py:139-152
full_pool = recon_pool                    # VQ-VAE round-trip of all 2149 real train crops
runs = {'B_pool': real + full_pool, ...}
```

`abc_recon.py` imports `VQVAE` and **never imports `PixelSNAIL`** — no prior is involved. So
the published "leaky" arm differs from the honest arm in **two** ways at once:

| | honest (`abc_budget`) | published "leaky" (`abc_recon` `B_pool`) | **true leaky (§7.2)** |
|---|---|---|---|
| pool contents | 360 **generated** samples | 2 149 **reconstructions of real crops** | 360 **generated** samples |
| generator | budget-*b* prior, from scratch | **none — no generator exists** | full-pool prior (`prior_b100`) |
| pool size | 360 | **2 149** | 360 |
| labels | conditioned class | true class of the real crop | conditioned class |

A +0.51 measured this way conflates three things: the leak, a ~6× larger training pool, and
the substitution of near-copies of real data for generated samples. It is a legitimate and
useful quantity — an **upper bound on what leakage could deliver if a generator reproduced its
training set faithfully** — but it is not what a leaky generator experiment reports.

## 7.2 The correct protocol, and how it was run

Generator trained on the full train pool, classifier restricted to budget *b*, pool size held
equal to the honest arm, filter unchanged.

`prior_b100_{top,bottom}_320.pt` is that generator — `warm=None`, `path=lmdb/train_pool_tight`
(all 4 274 train-pool crops; §3.3) — and `synth_b100/` is its 360-sample output. **No new code
was written**: `abc_budget.py` already parameterises exactly this contrast.

```bash
for B in 10 25 50; do
  CUDA_VISIBLE_DEVICES=1 python abc_budget.py \
    --budget $B --synth synth_b100 \
    --manifest results/manifest_tight_b${B}.csv \
    --out results/abc_leaky_b${B}.json
done
```

Everything except the pool's origin is identical to the honest run: same frozen subsample
(`build_train_items(rows, 1.0, seed=0)`), same 518-crop board-split test set, same ResNet-18
recipe, same 3 seeds, same no-leakage filter (the Condition-A model at budget *b*, seed *s*,
handed straight to `filter_pool`). Outputs: `results/abc_leaky_b{10,25,50}.json`.

## 7.3 Result

```bash
python3 -c "
import json, statistics as st
m=lambda p,k: st.mean([r[k]['macro_f1'] for r in json.load(open(p))])
for b in [10,25,50]:
    h,l=f'results/abc_budget_b{b}.json', f'results/abc_leaky_b{b}.json'
    print('b%-4d honest B-A %+.4f  C-A %+.4f   |   leaky B-A %+.4f  C-A %+.4f' %
          (b, m(h,'B')-m(h,'A'), m(h,'C')-m(h,'A'), m(l,'B')-m(l,'A'), m(l,'C')-m(l,'A')))
"
```

| budget | honest A | honest **B−A** | honest C−A | leaky A | leaky **B−A** | leaky C−A |
|---|---|---|---|---|---|---|
| 10% | 0.418598 | **+0.0123** | +0.0147 | 0.422248 | **+0.0020** | +0.0006 |
| 25% | 0.641112 | **−0.0457** | −0.0094 | 0.645408 | **−0.0659** | +0.0015 |
| 50% | 0.803530 | **−0.0442** | +0.0108 | 0.788738 | **−0.0168** | −0.0012 |

For comparison, the reconstruction bound the paper currently cites as leaky:
**+0.5100 / +0.2648 / +0.1212** at the same three budgets.

**The full-pool generator confers no benefit.** At 10% it gives **+0.002** where the published
figure is +0.510 — a factor of 255. At 25% and 50% the leaky pool *hurts*, as the honest pool
does. At 10% the honest generator is nominally *better* than the leaky one (+0.012 vs +0.002),
which is itself inside noise. Every leaky B−A and C−A here is within the ±0.015–0.025
run-to-run band established in §7.5.

## 7.4 Filter keep-rates — an independent corroboration

| budget | keep-rate, **honest** pool | keep-rate, **leaky** pool |
|---|---|---|
| 10% | 23.0% | 24.4% |
| 25% | **54.5%** | **28.9%** |
| 50% | **49.6%** | **31.4%** |

At 25% and 50% the budget-*b* filter rejects the **full-pool** generator's samples about twice
as often as it rejects that budget's own generator's samples. A pool carrying smuggled
information about the wider training set ought to look *more* like real data to a classifier,
not less. This reproduces, in the 6-way track, the effect §6.3 recorded in the binary track
(the filter rejects 64–95% of the full-pool generator's samples) and points the same way: the
full-pool prior's samples are further from the real distribution, not closer.

## 7.5 Run-to-run nondeterminism is larger than §"Two further observations" stated

Each leaky run retrains Condition A on data identical to the honest run's, with the same
seeds. The A columns are therefore a direct nondeterminism probe:

| budget | honest A | leaky-run A | \|diff\| |
|---|---|---|---|
| 10% | 0.418598 | 0.422248 | 0.003650 |
| 25% | 0.641112 | 0.645408 | 0.004295 |
| 50% | 0.803530 | 0.788738 | **0.014792** |

A second, cleaner probe already existed in the binary track and was missed until now: at the
100% budget `abc_binary.py:50` sets `honest = leaky`, so `B_honest` and `B_leaky` are two
trainings on **the identical pool with the identical seed** (confirmed: same `n`=360, same
keep-rate to 4 dp). Their differences are 0.003862, **0.025110**, 0.000958.

**The ±0.004 figure in §"Two further observations" is too optimistic.** Observed spread on
identical data and identical seeds reaches **0.015 (6-way)** and **0.025 (binary)**. The paper
should quote **±0.015, and up to ±0.025 on the binary task**. Consequences:

- The binary track's headline (+0.018 honest at 10%) sits at the edge of this band, and its
  leaky effects (+0.002 to +0.016) sit inside it.
- The 6-way harm figures (−0.046, −0.044, −0.052) remain well outside it.
- The b10 paired test (§2) is unaffected — pairing differences out is precisely what it does,
  and its own sd (0.031) already reflects this noise.

## 7.6 What this does to Finding 5c

The specific claim fails; the paper's central thesis survives and is better supported.

**Fails.** *"Replacing generated samples with reconstructions … lifts the 10% budget from
0.400 to 0.910, i.e. '+0.51 macro-F1 from synthetic data' … Any scarcity experiment whose
generator saw the full training set is measuring its own leak."* Run as an actual generator
experiment, a generator that saw the full training set delivers **+0.002**. The +0.51 is not
what leakage does here; it is what leakage *could* do if the generator worked.

**Survives, and is strengthened.** The gap between the two — +0.510 achievable versus +0.002
delivered — is a clean measurement of exactly what the paper argues: this pipeline's
class-conditional PixelSNAIL produces samples carrying almost none of the information its
training data holds. The reconstruction arm is the right control to keep; it is the ceiling.
The honest and true-leaky arms are the floor. The distance between ceiling and floor is the
result.

**Suggested reframing.** Report three arms, not two:

| arm | what it is | 10% budget |
|---|---|---|
| honest | budget-*b* generator, 360 samples | +0.012 |
| true leaky | full-pool generator, 360 samples | **+0.002** |
| reconstruction ceiling | VQ-VAE round-trip of the full pool, 2 149 crops | +0.510 |

and state the conclusion as: *a generator trained on the full training set buys nothing over
one trained on 10% of it, while merely round-tripping that training set through the same
decoder buys +0.51 — so the failure is in generation, not in what the generator was allowed to
see.* That is a stronger and more defensible claim than the one currently in Finding 5c, and
it retires the "any scarcity experiment whose generator saw the full training set is measuring
its own leak" sentence, which this experiment does not support.

## 7.7 Binary track — no re-run required

The binary track's leaky arm was **already** the correct protocol, verified in source:
`run_binary_track.sh:38-41` trains `uncond_tight` priors on `lmdb/train_pool_tight` (the full
pool) and samples `synth_binary_pool/`; `abc_binary.py:45-62` pairs that pool with each
budget's real subsample and the budget-*b* Condition-A filter.

```bash
python3 -c "
import json, statistics as st
d=json.load(open('results/abc_binary.json'))
for b in [0.1,0.25,0.5,1.0]:
    r=[x for x in d if abs(x['budget']-b)<1e-9]
    a=st.mean([x['A']['macro_f1'] for x in r])
    for c in ['B_honest','B_leaky']:
        print('%3d%% %-9s %+.4f' % (int(b*100), c, st.mean([x[c]['macro_f1'] for x in r])-a), end='   ')
    print()
"
```

| budget | honest B−A | **leaky B−A** |
|---|---|---|
| 10% | +0.0179 | **+0.0018** |
| 25% | +0.0074 | **+0.0155** |
| 50% | +0.0016 | **+0.0032** |
| 100% | +0.0013 | **−0.0081** (same pool as honest; difference is noise) |

Same picture as the 6-way track: the full-pool generator delivers nothing outside the noise
band, and at 100% the "two arms" are the same pool, so that row's −0.008 measures
nondeterminism, not leakage. `findings.md`'s existing sentence — *"even the leaky pool moves
nothing (+0.002 at 10%)"* — is correct and needs no change. The binary track therefore already
told the true story; only the 6-way track's leaky arm was mislabelled.

## §7 verdict

The leaky control, run correctly for the first time, gives **B−A = +0.002 / −0.066 / −0.017**
at the 10/25/50% budgets against a published +0.510 / +0.265 / +0.121 obtained from a
reconstruction substitute with a 6× larger pool. Finding 5c's specific claim is not supported.
The paper's central negative result is unaffected and, on the three-arm framing above, better
evidenced than before. Two further corrections follow from these runs: run-to-run
nondeterminism is **±0.015 (6-way) / ±0.025 (binary)**, not ±0.004; and the leaky pool is
rejected by the budget-*b* filter roughly twice as often as the honest pool at 25% and 50%.
