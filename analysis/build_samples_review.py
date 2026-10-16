"""samples_review.html — every generator's output, pixel-exact, no cherry-picking.

Sections
  1  conditional generators per budget (b10/b25/b50/b100): 6/class, captioned
     conditioned = X / classifier says = Y / filter kept-or-rejected
  2  binary-track generators (uncond_tight, uncond_b10, uncond_b25): 24 each, captioned with
     the defect/no-defect detector's verdict and confidence
  3  (woven into 1+2) nearest neighbour in the generator's OWN training crops, with cosine,
     against the calibrated null and ceiling; fraction-above-copy-threshold per generator
  4  summary strip per class: real | b100 | b10 | binary uncond_b10  — candidate paper figure

Honesty rules, enforced in code:
  * selection is random.Random(0).sample over each pool — stated on the page
  * "classifier says" = the tight 100% judge (classifier_Atight_b100_s0), the same judge used
    by every conditioning check, so page and numbers cannot disagree
  * "filter" = the budget-b Condition-A model, seed 0 — retrained here with the same
    set_seed(0) recipe as abc_budget, so decisions match the experiment. b100's filter IS
    classifier_Atight_b100_s0 (same manifest, budget, seed, recipe), reused not retrained.
  * NN similarity is reported against a null and ceiling COMPUTED IN THIS RUN, same embedding,
    same bank; the copy threshold is the 5th percentile of the ceiling distribution (i.e. 95%
    of literal copies score above it). Fractions use the FULL pool (360), not the 36 shown.
"""
import sys, pathlib  # repo layout (README.md): src/, experiments/, analysis/ importable from anywhere
sys.path[1:1] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('src', 'experiments', 'analysis')]
import argparse
import glob
import html
import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image

from pcb_utils import CLASSES, CLASS_TO_IDX, load_splits
from train_classifier import read_manifest, CropDS, build_train_items, build_test_items
from abc_recon import train_clf, TEST_TF

A = 'samples_review_assets'
SEED = 0
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])

COND = {10: 'synth_b10', 25: 'synth_b25', 50: 'synth_b50', 100: 'synth_b100'}
BINARY = {'uncond_tight (LEAKY: full train pool)': ('synth_binary_pool', 'perbbox'),
          'uncond_b10 (honest: 215 crops)': ('synth_binary_b10', 10),
          'uncond_b25 (honest: 537 crops)': ('synth_binary_b25', 25)}


def bank_paths(which):
    """The crops a given generator was actually trained on."""
    if which == 'perbbox':
        return [r['crop_path'] for r in read_manifest('results/manifest_tight_perbbox.csv')
                if r['split'] == 'train']
    return [r['crop_path'] for r in read_manifest(f'results/manifest_tight_b{which}.csv')
            if r['split'] == 'train']


@torch.no_grad()
def embed(paths, device, batch=64):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    m = m.to(device).eval()
    out = []
    for i in range(0, len(paths), batch):
        x = torch.stack([TF(Image.open(p).convert('RGB')) for p in paths[i:i + batch]]).to(device)
        out.append(torch.nn.functional.normalize(m(x), dim=1).cpu())
    del m
    torch.cuda.empty_cache()
    return torch.cat(out)


@torch.no_grad()
def load_clf(path, n_out, device):
    m = models.resnet18()
    m.fc = nn.Linear(m.fc.in_features, n_out)
    m.load_state_dict(torch.load(path, map_location='cpu'))
    return m.to(device).eval()


@torch.no_grad()
def predict(m, paths, device, batch=64):
    preds, confs = [], []
    for i in range(0, len(paths), batch):
        x = torch.stack([TF(Image.open(p).convert('RGB')) for p in paths[i:i + batch]]).to(device)
        pr = m(x).softmax(1)
        preds += pr.argmax(1).cpu().tolist()
        confs += pr.max(1).values.cpu().tolist()
    return preds, confs


def get_filter(budget, device):
    """The budget-b Condition-A model, seed 0 — identical recipe to abc_budget seed 0."""
    if budget == 100:
        return load_clf('checkpoint/classifier_Atight_b100_s0.pt', 6, device)
    ck = f'checkpoint/filter_b{budget}_s0.pt'
    man = f'results/manifest_tight_b{budget}.csv'
    rows = read_manifest(man)
    if not os.path.exists(ck):
        real = build_train_items(rows, 1.0, seed=0)
        vl = DataLoader(CropDS(build_test_items(rows), TEST_TF), batch_size=64, num_workers=8)
        print(f'  training b{budget} filter (A, seed 0, {len(real)} crops)...', flush=True)
        m, f1, _, _ = train_clf(real, vl, device, 0)
        torch.save(m.state_dict(), ck)
        print(f'    macro-F1 {f1:.3f} -> {ck}')
        return m.eval()
    return load_clf(ck, 6, device)


def tile_pair(gen_src, nn_src, cap_lines, bad=False):
    cap = '<br>'.join(cap_lines)
    return (f'<div class="pairbox{" bad" if bad else ""}"><div class="pair">'
            f'<figure class="t"><div class="imgwrap"><img src="{gen_src}"></div>'
            f'<figcaption>GENERATED</figcaption></figure>'
            f'<figure class="t"><div class="imgwrap"><img src="{nn_src}"></div>'
            f'<figcaption>nearest TRAIN</figcaption></figure></div>'
            f'<div class="cap">{cap}</div></div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per_class', type=int, default=6)
    ap.add_argument('--n_binary', type=int, default=24)
    args = ap.parse_args()
    device = 'cuda'
    rng = random.Random(SEED)
    for d in ('gen', 'nn', 'real'):
        os.makedirs(f'{A}/{d}', exist_ok=True)

    judge = load_clf('checkpoint/classifier_Atight_b100_s0.pt', 6, device)
    detector = load_clf('checkpoint/defect_detector.pt', 2, device)

    # ---- embed every distinct bank path ONCE ----
    # b100 has no subsample manifest — its generator trained on the full perbbox train pool
    banks = {b: bank_paths(b) for b in [10, 25, 50, 'perbbox']}
    banks[100] = banks['perbbox']
    all_paths = sorted({p for v in banks.values() for p in v})
    print(f'embedding {len(all_paths)} distinct training crops...', flush=True)
    all_emb = embed(all_paths, device)
    idx = {p: i for i, p in enumerate(all_paths)}

    def bank_mat(which):
        ps = banks[which]
        return ps, all_emb[[idx[p] for p in ps]]

    # ---- null and ceiling, same embedding, same bank ----
    test_paths = [r['crop_path'] for r in read_manifest('results/manifest_tight_perbbox.csv')
                  if r['split'] == 'test']
    null_q = rng.sample(test_paths, 120)
    ceil_q = [os.path.join('recon_pool', os.path.basename(p))
              for p in rng.sample(banks['perbbox'], 200)]
    ceil_q = [p for p in ceil_q if os.path.exists(p)][:120]
    B = all_emb[[idx[p] for p in banks['perbbox']]].to(device)
    nullv = (embed(null_q, device).to(device) @ B.T).max(1).values.cpu().numpy()
    ceilv = (embed(ceil_q, device).to(device) @ B.T).max(1).values.cpu().numpy()
    COPY_T = float(np.percentile(ceilv, 5))
    print(f'null (novel real) {nullv.mean():.3f} | ceiling (copy) {ceilv.mean():.3f} | '
          f'copy threshold (5th pct of ceiling) {COPY_T:.3f}')

    stats = {'null': float(nullv.mean()), 'ceiling': float(ceilv.mean()), 'copy_t': COPY_T,
             'generators': {}}
    sections = {}

    # ================= conditional generators =================
    for budget, pool_dir in COND.items():
        pool = sorted(glob.glob(f'{pool_dir}/*.png'))
        if not pool:
            print(f'  b{budget}: pool missing, skipped'); continue
        bank_ps, bank_e = bank_mat(budget if budget != 100 else 100)
        bank_e = bank_e.to(device)

        # full-pool stats (all 360)
        emb_all = embed(pool, device).to(device)
        sims_all, nn_all = (emb_all @ bank_e.T).max(1)
        sims_all = sims_all.cpu().numpy()
        frac_copy = float((sims_all > COPY_T).mean())

        filt = get_filter(budget, device)
        sel = []
        for c in CLASSES:
            cls_pool = [p for p in pool if os.path.basename(p).rsplit('_', 1)[0] == c]
            sel += rng.sample(cls_pool, min(args.per_class, len(cls_pool)))
        jp, jc = predict(judge, sel, device)
        fp, _ = predict(filt, sel, device)
        del filt
        torch.cuda.empty_cache()

        cards = []
        for k, p in enumerate(sel):
            cls = os.path.basename(p).rsplit('_', 1)[0]
            i = pool.index(p)
            sim = float(sims_all[i])
            nn_p = bank_ps[int(nn_all[i])]
            g = f'{A}/gen/b{budget}_{os.path.basename(p)}'
            n = f'{A}/nn/b{budget}_{os.path.basename(p)}'
            Image.open(p).save(g)
            Image.open(nn_p).convert('RGB').save(n)
            kept = CLASSES[fp[k]] == cls
            good = CLASSES[jp[k]] == cls
            scls = 'no' if sim > COPY_T else 'ok'
            cards.append((cls, tile_pair(g, n, [
                f'conditioned = <b>{cls}</b>',
                f'classifier says = <b class="{"ok" if good else "no"}">{CLASSES[jp[k]]}</b> '
                f'({100*jc[k]:.0f}%)',
                f'filter: <b class="{"ok" if kept else "no"}">{"KEPT" if kept else "REJECTED"}</b>',
                f'NN cos <b class="{scls}">{sim:.3f}</b>'], bad=not good)))
        sections[f'b{budget}'] = cards
        stats['generators'][f'cond_b{budget}'] = {
            'n_pool': len(pool), 'mean_nn': float(sims_all.mean()),
            'frac_above_copy_threshold': frac_copy}
        print(f'  cond b{budget}: NN mean {sims_all.mean():.3f}, '
              f'{100*frac_copy:.1f}% above copy threshold', flush=True)

    # ================= binary generators =================
    bin_cards = {}
    for label, (pool_dir, bank_key) in BINARY.items():
        pool = sorted(glob.glob(f'{pool_dir}/*.png'))
        if not pool:
            continue
        bank_ps, bank_e = bank_mat(bank_key)
        bank_e = bank_e.to(device)
        emb_all = embed(pool, device).to(device)
        sims_all, nn_all = (emb_all @ bank_e.T).max(1)
        sims_all = sims_all.cpu().numpy()
        frac_copy = float((sims_all > COPY_T).mean())

        sel = rng.sample(pool, min(args.n_binary, len(pool)))
        dp, dc = predict(detector, sel, device)
        cards = []
        for k, p in enumerate(sel):
            i = pool.index(p)
            sim = float(sims_all[i])
            tag = label.split(' ')[0]
            g = f'{A}/gen/{tag}_{os.path.basename(p)}'
            n = f'{A}/nn/{tag}_{os.path.basename(p)}'
            Image.open(p).save(g)
            Image.open(bank_ps[int(nn_all[i])]).convert('RGB').save(n)
            has = dp[k] == 1
            scls = 'no' if sim > COPY_T else 'ok'
            cards.append(tile_pair(g, n, [
                f'detector: <b class="{"ok" if has else "no"}">'
                f'{"defect" if has else "NO defect"}</b> (p={dc[k]:.2f})',
                f'NN cos <b class="{scls}">{sim:.3f}</b>'], bad=not has))
        bin_cards[label] = (cards, [dp[k] == 1 for k in range(len(sel))])
        stats['generators'][pool_dir] = {
            'n_pool': len(pool), 'mean_nn': float(sims_all.mean()),
            'frac_above_copy_threshold': frac_copy,
            'detector_defect_rate_shown': float(np.mean(dp))}
        print(f'  {pool_dir}: NN mean {sims_all.mean():.3f}, '
              f'{100*frac_copy:.1f}% above copy threshold', flush=True)

    # ================= section 4: summary strips =================
    tight = read_manifest('results/manifest_tight.csv')
    strips = []
    b100_pool = sorted(glob.glob(f'{COND[100]}/*.png'))
    b10_pool = sorted(glob.glob(f'{COND[10]}/*.png'))
    bin10 = sorted(glob.glob('synth_binary_b10/*.png'))
    for c in CLASSES:
        real_p = rng.choice([r['crop_path'] for r in tight
                             if r['class_name'] == c and r['split'] == 'train'])
        cells = [('REAL', real_p)]
        for name, pl in (('cond b100', b100_pool), ('cond b10', b10_pool)):
            cp = [p for p in pl if os.path.basename(p).rsplit('_', 1)[0] == c]
            cells.append((name, rng.choice(cp) if cp else None))
        cells.append(('binary uncond_b10 (no class exists)', rng.choice(bin10) if bin10 else None))
        row = []
        for name, p in cells:
            if p is None:
                continue
            dst = f'{A}/real/strip_{c}_{name.split()[0]}_{os.path.basename(p)}'
            Image.open(p).convert('RGB').save(dst)
            row.append((name, dst))
        strips.append((c, row))

    write_html(sections, bin_cards, strips, stats, args)
    json.dump(stats, open('results/samples_review.json', 'w'), indent=2)
    print('\nwrote samples_review.html')


def write_html(sections, bin_cards, strips, st, args):
    g = st['generators']
    P = [f'''<!doctype html><meta charset="utf-8"><title>Samples review — every generator</title>
<style>
body{{margin:0;padding:30px;background:#111417;color:#e8e8e8;font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}}
h1{{margin:0 0 6px}} h2{{margin:40px 0 6px;font-size:20px;border-bottom:1px solid #2a2f35;padding-bottom:8px}}
h3{{margin:22px 0 8px;font-size:14px;color:#7fd1ff;font-family:ui-monospace,monospace}}
.note{{color:#9aa3ab;max-width:1000px}}
.grid{{display:flex;flex-wrap:wrap;gap:12px}}
.pairbox{{background:#1a1e22;border:1px solid #2a2f35;border-radius:6px;padding:8px}}
.pairbox.bad{{border-color:#7a2a2a}}
.pair{{display:flex;gap:6px}}
figure.t{{margin:0}} .imgwrap{{width:256px;height:256px;overflow:hidden}}
.imgwrap img{{display:block;width:256px;height:256px;image-rendering:pixelated}}
figcaption{{font:10px ui-monospace,monospace;color:#9aa3ab;margin-top:3px}}
.cap{{margin-top:6px;font:11px/1.5 ui-monospace,monospace}}
.ok{{color:#4ade80}} .no{{color:#ff6b6b}}
table{{border-collapse:collapse;margin:12px 0}}
td,th{{border:1px solid #2a2f35;padding:6px 12px;font:12px ui-monospace,monospace;text-align:right}}
th{{color:#9aa3ab;text-align:left}} td.hi{{background:#3a1414;color:#ff9b9b}}
.strip{{display:flex;gap:8px;margin:8px 0 18px;flex-wrap:wrap}}
.strip figure{{margin:0}} .strip figcaption{{font:11px ui-monospace,monospace;color:#cbd5e1;margin-top:4px;max-width:256px}}
</style>
<h1>Samples review — see what every generator actually makes</h1>
<p class="note">All images 256×256 at 256×256, <code>image-rendering:pixelated</code>. Selection is
<code>random.Random({SEED}).sample</code> over each full pool — nothing curated. Every sample is
shown beside its <b>nearest neighbour among the crops that generator was trained on</b> (cosine,
ImageNet-ResNet-18). Calibration computed in this run, same embedding, same bank:
<b>null {st["null"]:.3f}</b> (a never-seen real crop), <b>ceiling {st["ceiling"]:.3f}</b> (a VQ-VAE
recon of a training crop = a literal copy). <b>Copy threshold {st["copy_t"]:.3f}</b> = 5th
percentile of the ceiling — 95% of true copies score above it.</p>
<h2>0 · Copy-rate summary (computed over each FULL pool, not just the samples shown)</h2>
<table><tr><th>generator</th><th>pool</th><th>mean NN cos</th><th>above copy threshold</th></tr>''']
    for k, v in g.items():
        hi = ' class="hi"' if v['frac_above_copy_threshold'] > 0.10 else ''
        P.append(f'<tr><th>{k}</th><td>{v["n_pool"]}</td><td>{v["mean_nn"]:.3f}</td>'
                 f'<td{hi}>{100*v["frac_above_copy_threshold"]:.1f}%</td></tr>')
    P.append('</table>')

    P.append('<h2>1 · Conditional generators, per budget — 6 per class</h2>')
    P.append('<p class="note">Caption per sample: the class it was <b>conditioned</b> on; what the '
             'tight 100% judge <b>says</b> it is; whether the budget-b <b>filter</b> (the '
             'Condition-A model at that budget, seed 0 — the no-leakage rule) kept it; and its '
             'nearest-neighbour cosine. Red border = judge disagrees with the conditioning.</p>')
    for tag, cards in sections.items():
        P.append(f'<h2>1·{tag} — conditional generator</h2>')
        for c in CLASSES:
            cc = [h for cls, h in cards if cls == c]
            if not cc:
                continue
            P.append(f'<h3>conditioned on: {c}</h3><div class="grid">')
            P.extend(cc)
            P.append('</div>')

    P.append('<h2>2 · Binary-track generators — 24 samples each</h2>')
    P.append('<p class="note">No class exists for these (unconditional priors, n_img_class=0). '
             'Caption: the defect/no-defect detector\'s verdict and confidence (held-out accuracy '
             '95.0%). Red border = detector sees no defect — a failed "synthetic positive".</p>')
    for label, (cards, verdicts) in bin_cards.items():
        rate = 100 * np.mean(verdicts)
        P.append(f'<h3>{html.escape(label)} — detector says "defect" in {rate:.0f}% of shown</h3>')
        P.append('<div class="grid">')
        P.extend(cards)
        P.append('</div>')

    P.append('<h2>4 · Summary strip per class — candidate paper figure</h2>')
    P.append('<p class="note">Left to right: a real tight crop of the class · what the b100 '
             'conditional generator makes when asked for it · what the b10 conditional generator '
             'makes · a binary-track sample (unconditional — no class was or could be requested).</p>')
    for c, row in strips:
        P.append(f'<h3>{c}</h3><div class="strip">')
        for name, p in row:
            P.append(f'<figure><div class="imgwrap"><img src="{p}"></div>'
                     f'<figcaption>{html.escape(name)}</figcaption></figure>')
        P.append('</div>')
    open('samples_review.html', 'w').write('\n'.join(P) + '\n')


if __name__ == '__main__':
    main()
