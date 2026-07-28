# AUDIT — adversarial verification of five claims

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
