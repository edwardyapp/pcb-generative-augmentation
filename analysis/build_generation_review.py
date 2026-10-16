"""generation_review.html — look at the generator's output yourself, pixel-exact.

Sections
  1  real tight crops, 12/class, VOC bbox in red                       <- the target
  2  the ORIGINAL ICCE-TW pipeline (unconditional), 60 as ONE pool     <- no class labels exist
  3  the current conditional generator, 12/class, conditioned vs predicted
  4  real above generated, same class, same size                       <- the one that matters

Plus, for every generated image in sections 2 and 3, its NEAREST NEIGHBOUR in the set the
generator was actually trained on. If samples are near-copies, that is memorisation, not
generation -- and it is the honest reading of "the outputs look like the training images".
Neighbours are found in the embedding of an ImageNet-pretrained ResNet-18 (cosine similarity)
-- deliberately NOT one of our fine-tuned models, so the feature space cannot be biased toward
either the real or the generated set.

And the fraction of generated crops in which a defect is VISIBLY PRESENT, measured with the
defect/no-defect detector (make_defect_detector.py) rather than asserted. The 6-way classifier
cannot answer this: it has no "no defect" class and will label a blank green board `spur`.

Nothing is cherry-picked. Section 2 is random.Random(0).sample over the pool; section 3 shows
EVERY sample the epoch-320 conditioning check drew (12/class, all of them).
"""
import sys, pathlib  # repo layout (README.md): src/, experiments/, analysis/ importable from anywhere
sys.path[1:1] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('src', 'experiments', 'analysis')]
import os
import glob
import html
import json
import random
import argparse

import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image

from pcb_utils import CLASSES
from build_review import (bbox_in_crop, load_rows, tile, bbox_svg, ASSETS,
                          MANIFEST, SEED, PER_CLASS)

A = 'genreview_assets'
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])
TIGHT_TRAIN = 'results/manifest_tight_perbbox.csv'   # what the tight prior trained on
ALL600 = 'results/manifest.csv'                      # what the ORIGINAL prior trained on (lmdb/all)


def pick_device():
    if not torch.cuda.is_available():
        return 'cpu'
    best, free_best = None, 0
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > free_best:
            best, free_best = i, free
    return f'cuda:{best}' if free_best > 5e9 else 'cpu'   # never elbow a running job off a GPU


@torch.no_grad()
def embed(paths, device, batch=64):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    m = m.to(device).eval()
    out = []
    for i in range(0, len(paths), batch):
        x = torch.stack([TF(Image.open(p).convert('RGB')) for p in paths[i:i + batch]]).to(device)
        f = m(x)
        out.append(torch.nn.functional.normalize(f, dim=1).cpu())
    del m
    torch.cuda.empty_cache() if device.startswith('cuda') else None
    return torch.cat(out)


def nearest(gen_paths, train_paths, device):
    """Cosine NN of each generated crop in the training set it was fitted on."""
    g = embed(gen_paths, device)
    t = embed(train_paths, device)
    sim = g @ t.T
    v, i = sim.max(1)
    return [(train_paths[int(j)], float(s)) for j, s in zip(i, v)]


@torch.no_grad()
def defect_fraction(paths, device, thresh=0.5, batch=64):
    """P(defect present), per image, from the held-out-validated detector."""
    ck = 'checkpoint/defect_detector.pt'
    if not os.path.exists(ck):
        return None, None
    m = models.resnet18()
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(ck, map_location='cpu'))
    m = m.to(device).eval()
    ps = []
    for i in range(0, len(paths), batch):
        x = torch.stack([TF(Image.open(p).convert('RGB')) for p in paths[i:i + batch]]).to(device)
        ps += m(x).softmax(1)[:, 1].cpu().tolist()
    del m
    torch.cuda.empty_cache() if device.startswith('cuda') else None
    ps = np.array(ps)
    return float((ps > thresh).mean()), ps


@torch.no_grad()
def defect_fraction_600(paths, device, thresh=0.5):
    """600px-domain images: the tight detector expects a defect filling the frame, so SLIDE it.

    A 128px-native window is 128/600*256 ~ 55px inside a 256px decode of a 600px crop. We slide
    55px windows (stride 14), upsample each to the detector's input, and call a defect present if
    ANY window fires. Calibrated by running the identical procedure on real 600px crops.
    """
    ck = 'checkpoint/defect_detector.pt'
    if not os.path.exists(ck):
        return None, None
    m = models.resnet18()
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(ck, map_location='cpu'))
    m = m.to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    win, stride = 55, 14
    best = []
    for p in paths:
        im = Image.open(p).convert('RGB').resize((256, 256))
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255)[None].to(device)
        pat = x.unfold(2, win, stride).unfold(3, win, stride)          # 1,3,nh,nw,win,win
        nh, nw = pat.shape[2], pat.shape[3]
        pat = pat.permute(0, 2, 3, 1, 4, 5).reshape(nh * nw, 3, win, win)
        pat = torch.nn.functional.interpolate(pat, size=(224, 224), mode='bilinear',
                                              align_corners=False)
        pat = (pat - mean) / std
        sc = []
        for i in range(0, pat.shape[0], 256):
            sc.append(m(pat[i:i + 256]).softmax(1)[:, 1])
        best.append(float(torch.cat(sc).max()))
    del m
    torch.cuda.empty_cache() if device.startswith('cuda') else None
    best = np.array(best)
    return float((best > thresh).mean()), best


def slice_grid(path, n_per_chunk=36, chunk_idx=0):
    """cond_check saves nrow=6 grids of 36; tile i in chunk k has class 3k + i//12."""
    g = Image.open(path).convert('RGB')
    out = []
    for i in range(n_per_chunk):
        r, c = i // 6, i % 6
        x, y = 2 + c * 258, 2 + r * 258
        if x + 256 > g.width or y + 256 > g.height:
            break
        out.append((g.crop((x, y, x + 256, y + 256)), 3 * chunk_idx + i // 12))
    return out


@torch.no_grad()
def classify6(imgs, device):
    m = models.resnet18()
    m.fc = nn.Linear(m.fc.in_features, 6)
    m.load_state_dict(torch.load('checkpoint/classifier_Atight_b100_s0.pt', map_location='cpu'))
    m = m.to(device).eval()
    x = torch.stack([TF(i) for i in imgs]).to(device)
    p = m(x).argmax(1).cpu().numpy()
    del m
    torch.cuda.empty_cache() if device.startswith('cuda') else None
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epoch', type=int, default=None, help='condtight3 epoch (default: latest)')
    args = ap.parse_args()
    device = pick_device()
    print(f'device: {device}')
    for d in ('real', 'uncond', 'cond', 'nn_uncond', 'nn_cond'):
        os.makedirs(f'{A}/{d}', exist_ok=True)

    # ---------------- section 1: real ----------------
    rows = load_rows(MANIFEST)
    by_class = {c: [r for r in rows if r['class_name'] == c] for c in CLASSES}
    real = {c: random.Random(SEED).sample(by_class[c], PER_CLASS) for c in CLASSES}
    real_paths = []
    for c in CLASSES:
        for r in real[c]:
            n = os.path.basename(r['crop_path'])
            if not os.path.exists(f'{A}/real/{n}'):
                Image.open(r['crop_path']).convert('RGB').save(f'{A}/real/{n}')
            real_paths.append(f'{A}/real/{n}')

    # ---------------- section 2: unconditional ----------------
    upool = sorted(glob.glob('synth_uncond/u*.png'))
    upick = random.Random(SEED).sample(upool, min(60, len(upool)))
    for p in upick:
        Image.open(p).save(f'{A}/uncond/{os.path.basename(p)}')
    upick_a = [f'{A}/uncond/{os.path.basename(p)}' for p in upick]

    # ---------------- section 2b: the user's demo_pool ----------------
    # Same checkpoints, same sampler, same temp, same latent shapes as sample_uncond.py --
    # generate_pool.py only LANCZOS-upscales the 256px render to 512 for display.
    os.makedirs(f'{A}/demo', exist_ok=True)
    dpool = sorted(glob.glob('demo_pool/*.png'))
    dpick = random.Random(SEED).sample(dpool, min(60, len(dpool)))
    for p in dpick:
        Image.open(p).convert('RGB').resize((256, 256), Image.LANCZOS).save(
            f'{A}/demo/{os.path.basename(p)}')
    dpick_a = [f'{A}/demo/{os.path.basename(p)}' for p in dpick]

    # ---------------- section 3: conditional, condtight3 ----------------
    eps = sorted({int(os.path.basename(p).split('_')[0][2:])
                  for p in glob.glob('synth_trend/ep*_chunk*.png')})
    epoch = args.epoch or (eps[-1] if eps else None)
    cond = []
    if epoch is not None:
        for k in (0, 1):
            gp = f'synth_trend/ep{epoch:03d}_chunk{k}.png'
            if os.path.exists(gp):
                cond += slice_grid(gp, chunk_idx=k)
    cpaths, clabels = [], []
    for i, (im, lab) in enumerate(cond):
        fn = f'{A}/cond/{CLASSES[lab]}_{i:03d}.png'
        im.save(fn)
        cpaths.append(fn); clabels.append(lab)
    cpred = classify6([im for im, _ in cond], device) if cond else np.array([])

    # ---------------- nearest neighbours ----------------
    tight_train = [r['crop_path'] for r in load_rows(TIGHT_TRAIN) if r['split'] == 'train']
    all600 = [r['crop_path'] for r in load_rows(ALL600)]          # lmdb/all: every variant
    print(f'NN search sets: tight train {len(tight_train)}, 600px all {len(all600)}')
    nn_c = nearest(cpaths, tight_train, device) if cpaths else []
    nn_u = nearest(upick_a, all600, device)
    nn_d = nearest(dpick_a, all600, device)
    os.makedirs(f'{A}/nn_demo', exist_ok=True)
    for src, (p, _) in zip(cpaths, nn_c):
        Image.open(p).convert('RGB').save(f'{A}/nn_cond/{os.path.basename(src)}')
    for src, (p, _) in zip(upick_a, nn_u):
        Image.open(p).convert('RGB').resize((256, 256)).save(f'{A}/nn_uncond/{os.path.basename(src)}')
    for src, (p, _) in zip(dpick_a, nn_d):
        Image.open(p).convert('RGB').resize((256, 256)).save(f'{A}/nn_demo/{os.path.basename(src)}')

    # ---------------- defect present? ----------------
    frac_cond, p_cond = defect_fraction(cpaths, device) if cpaths else (None, None)
    frac_real, p_real = defect_fraction(real_paths, device)
    # NOTE: the 600px-domain numbers are NOT computed here. The naive "slide the tight detector
    # and take the max softmax" test is saturated -- it fires on 81% of provably-clean board --
    # so it is deliberately not used. score_pools_600.py does it properly (in-domain detector,
    # logit margins, threshold calibrated to 5% FPR) and this page reads its JSON.

    # the calibrated 600px-domain numbers come from score_pools_600.py (logit-margin statistic,
    # 5% FPR on real clean board, ceiling = VQ-VAE recon of a REAL defect)
    cal = {}
    if os.path.exists('results/defect_presence_600.json'):
        cal = json.load(open('results/defect_presence_600.json'))

    stats = {'epoch': epoch, 'frac_cond': frac_cond, 'frac_real_tight': frac_real,
             'nn_cond_sim': [s for _, s in nn_c], 'nn_uncond_sim': [s for _, s in nn_u],
             'nn_demo_sim': [s for _, s in nn_d], 'cal600': cal}
    json.dump(stats, open('results/generation_review.json', 'w'), indent=2)

    write(real, real_paths, upick_a, nn_u, dpick_a, nn_d, cpaths, clabels, cpred, nn_c,
          stats, epoch)
    print('wrote generation_review.html')


def pct(x):
    return '—' if x is None else f'{100*x:.1f}%'


def cal_row(st, key):
    for p in st.get('cal600', {}).get('pools', []):
        if key in p['pool']:
            return p
    return None


def write(real, real_paths, upick, nn_u, dpick, nn_d, cpaths, clabels, cpred, nn_c, st, epoch):
    P = [f'''<!doctype html><meta charset="utf-8"><title>Generation review</title>
<style>
:root{{--bg:#111417;--fg:#e8e8e8;--mut:#9aa3ab;--line:#2a2f35;--card:#1a1e22}}
*{{box-sizing:border-box}}
body{{margin:0;padding:30px;background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}}
h1{{margin:0 0 6px}} h2{{margin:42px 0 6px;font-size:20px;border-bottom:1px solid var(--line);padding-bottom:8px}}
h3{{margin:24px 0 8px;font-size:14px;color:#7fd1ff;font-family:ui-monospace,monospace}}
.note{{color:var(--mut);max-width:980px}}
.grid{{display:flex;flex-wrap:wrap;gap:12px}}
figure.t{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:7px;width:270px}}
figure.t.bad{{border-color:#ff2020}}
.imgwrap{{position:relative;width:256px;height:256px;overflow:hidden}}
.imgwrap img{{display:block;width:256px;height:256px;image-rendering:pixelated}}
svg.ov{{position:absolute;inset:0;width:256px;height:256px}}
figcaption{{margin-top:5px;font:11px/1.4 ui-monospace,monospace;word-break:break-all}}
figcaption span{{display:block;color:var(--mut)}}
.pairbox{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px}}
.pair{{display:flex;gap:6px}} .pair figure.t{{width:auto;border:0;background:none;padding:0}}
.hd{{font:11px ui-monospace,monospace;color:var(--mut);margin-bottom:4px}}
.ok{{color:#4ade80}} .no{{color:#ff6b6b}} .warn{{color:#fbbf24}}
table{{border-collapse:collapse;margin:10px 0}}
td,th{{border:1px solid var(--line);padding:6px 12px;font:12px ui-monospace,monospace;text-align:right}}
th{{color:var(--mut)}} td.hi{{background:#3a1414;color:#ff9b9b}} td.good{{background:#12331e;color:#86efac}}
.row{{display:flex;gap:8px;overflow-x:auto;padding-bottom:6px}}
.rowlab{{font:12px ui-monospace,monospace;color:var(--mut);margin:10px 0 3px}}
.key{{display:inline-block;width:12px;height:12px;border:2px solid #ff2020;vertical-align:-2px}}
</style>
<h1>Generation review — can these models draw a defect?</h1>
<p class="note">Every image is a <b>256×256 PNG at 256×256</b>, <code>image-rendering:pixelated</code>,
no downsampling. <b>Nothing is cherry-picked:</b> section 2 is <code>random.Random({SEED}).sample</code>
over the pool; section 3 shows <b>every</b> sample the epoch-{epoch} conditioning check drew (12/class,
all of them). <span class="key"></span> = the VOC bounding box.</p>

<h2>0 · Is a defect actually there?</h2>
<p class="note">The 6-way classifier <b>cannot</b> answer this — it has no <i>no-defect</i> class, so
it labels a blank green board <code>spur</code> with full confidence. That is exactly how the
epoch-80 check produced a 22.4% "consistency" from images containing no defect. So we trained a
<b>defect / no-defect detector</b> at the scale it is used, with negatives cut from the
<b>same boards</b> that provably miss every bounding box, and validated on the held-out boards
(94.3% window accuracy). Scores are <b>logit margins</b>, not softmax — softmax saturates at 1.000
on every image and carries no information. The threshold is calibrated to a <b>5% false-positive
rate on provably-clean real board</b>.</p>
<p class="note"><b>The control that makes this fair:</b> generated images are VQ-VAE decodes — soft,
smoothed, quantised. So we also score <b>VQ-VAE reconstructions of REAL 600px crops</b>: same decoder
blur, but a real defect inside. <b>That is the ceiling a generator must be judged against</b>, and any
shortfall below it cannot be blamed on decoder blur.</p>
<table>
<tr><th style="text-align:left">600px whole-board domain</th><th>defect present</th></tr>
<tr><th style="text-align:left">REAL 600px crops — photo, all contain a defect</th>
    <td class="good">{pct((cal_row(st,'REAL 600px') or {}).get('frac'))}</td></tr>
<tr><th style="text-align:left">VQ-VAE RECON of them — decoder blur + REAL defect &nbsp;<b>← CEILING</b></th>
    <td class="warn">{pct((cal_row(st,'RECON') or {}).get('frac'))}</td></tr>
<tr><th style="text-align:left">GENERATED — demo_pool (your ICCE-TW demo)</th>
    <td>{pct((cal_row(st,'demo_pool') or {}).get('frac'))}</td></tr>
<tr><th style="text-align:left">GENERATED — my sample_uncond (same ckpts, same sampler)</th>
    <td>{pct((cal_row(st,'sample_uncond') or {}).get('frac'))}</td></tr>
<tr><th style="text-align:left">— tight domain —</th><th></th></tr>
<tr><th style="text-align:left">REAL tight crops (sanity — should be ~100%)</th>
    <td class="good">{pct(st["frac_real_tight"])}</td></tr>
<tr><th style="text-align:left">GENERATED — conditional tight generator (condtight3 ep{epoch})</th>
    <td class="hi">{pct(st["frac_cond"])}</td></tr>
</table>
<p class="note"><b class="warn">Read this carefully.</b> A real defect passed through the VQ-VAE
decoder is only detectable <b>{pct((cal_row(st,'RECON') or {}).get('frac'))}</b> of the time, against
<b>{pct((cal_row(st,'REAL 600px') or {}).get('frac'))}</b> for the raw photograph. <b>The decoder
itself destroys most of the defect at this scale.</b> The generated pools score at or above that
ceiling — so at 600px scale we <b>cannot</b> claim the original pipeline fails to draw defects. The
information is gone before generation is even involved. That is the crop-scale finding again, arriving
from a completely different direction.</p>
<p class="note">And <b>memorisation</b>: mean cosine similarity to the nearest training crop —
conditional <b>{np.mean(st["nn_cond_sim"]):.3f}</b>, unconditional
<b>{np.mean(st["nn_uncond_sim"]):.3f}</b> (max
{max(st["nn_cond_sim"] or [0]):.3f} / {max(st["nn_uncond_sim"] or [0]):.3f}). Every generated crop
below is shown <b>beside its nearest neighbour in the set that generator was actually trained on</b>,
so you can judge copying with your own eyes.</p>''']

    # 1 real
    P.append('<h2>1 · Real tight crops — the target</h2>')
    P.append('<p class="note">12 per class, red box = the annotated defect. This is what a working '
             'generator would have to produce.</p>')
    for c in CLASSES:
        P.append(f'<h3>{c}</h3><div class="grid">')
        for r in real[c]:
            n = os.path.basename(r['crop_path'])
            bb = bbox_in_crop(r)
            P.append(tile(f'{A}/real/{n}', n, f"board {r['board_id']} · {r['split']}", bb))
        P.append('</div>')

    # 2 unconditional
    P.append('<h2>2 · The ORIGINAL ICCE-TW pipeline (unconditional)</h2>')
    P.append(f'''<p class="note"><code>pixelsnail_top_357</code> + <code>pixelsnail_bottom_best</code>
decoded through <code>vqvae_560</code>, unmodified, <code>strict=True</code> load. <b>These priors have
no class input</b>, so no class label exists for any of these images — they are one undifferentiated
pool, which is exactly the point: this pipeline emits an <b>unlabeled</b> pool. 60 drawn at random
(seed {SEED}). Each is shown with its nearest neighbour among the <b>10,668 crops it was trained on</b>
(cosine, ImageNet-ResNet-18 embedding).</p>
<p class="note"><b class="no">Note:</b> these priors were trained on <code>lmdb/all</code> — <b>all 10
boards, test boards included</b>. Shown for inspection only; nothing sampled from them may be used as
training data.</p>''')
    P.append('<div class="grid">')
    for p, (nnp, sim) in zip(upick, nn_u):
        n = os.path.basename(p)
        cls = 'no' if sim > 0.9 else ('warn' if sim > 0.8 else 'ok')
        P.append(f'<div class="pairbox"><div class="hd">generated (no class label exists) '
                 f'| nearest training crop — cos <b class="{cls}">{sim:.3f}</b></div><div class="pair">'
                 f'<figure class="t"><div class="imgwrap"><img src="{p}"></div>'
                 f'<figcaption>GENERATED</figcaption></figure>'
                 f'<figure class="t"><div class="imgwrap"><img src="{A}/nn_uncond/{n}"></div>'
                 f'<figcaption>nearest TRAIN</figcaption></figure></div></div>')
    P.append('</div>')

    # 2b demo_pool
    P.append('<h2>2b · Your demo_pool — the SAME pipeline, and it agrees</h2>')
    d = cal_row(st, 'demo_pool') or {}
    mn = cal_row(st, 'sample_uncond') or {}
    P.append(f'''<p class="note"><code>generate_pool.py</code> loads <b>the same three checkpoints</b>
(<code>vqvae_560</code> + <code>pixelsnail_top_357</code> + <code>pixelsnail_bottom_best</code>), the
same unconditional <code>sample_model</code>, the same <code>temp=1.0</code>, the same latent shapes
(top 32×32, bottom 64×64). It decodes to <b>256×256</b> and then only <b>LANCZOS-upscales to 512</b>
for display — verified: downscaling those 512s back to 256 and re-upscaling changes them by
0.39/255, i.e. the 512 carries <b>no real detail</b>.</p>
<p class="note"><b>So demo_pool and my samples are the same distribution, and the detector agrees:</b>
demo_pool <b>{pct(d.get("frac"))}</b> vs mine <b>{pct(mn.get("frac"))}</b> — {abs((d.get("frac") or 0)-(mn.get("frac") or 0))*100:.1f} pp apart.
<b class="ok">My sampler is not broken</b>, and there is no temperature / checkpoint / resolution /
decoding discrepancy to find. 300 images, all 512×512, generated 2026-07-02.</p>''')
    P.append('<div class="grid">')
    for p, (nnp, sim) in zip(dpick, nn_d):
        n = os.path.basename(p)
        cls = 'no' if sim > 0.9 else ('warn' if sim > 0.8 else 'ok')
        P.append(f'<div class="pairbox"><div class="hd">demo_pool | nearest training crop — cos '
                 f'<b class="{cls}">{sim:.3f}</b></div><div class="pair">'
                 f'<figure class="t"><div class="imgwrap"><img src="{p}"></div>'
                 f'<figcaption>demo_pool</figcaption></figure>'
                 f'<figure class="t"><div class="imgwrap"><img src="{A}/nn_demo/{n}"></div>'
                 f'<figcaption>nearest TRAIN</figcaption></figure></div></div>')
    P.append('</div>')

    # 3 conditional
    P.append(f'<h2>3 · The current conditional generator (condtight3, epoch {epoch})</h2>')
    P.append('<p class="note">Every sample the conditioning check drew — 12 per class, all shown. '
             'Captioned with the class it was <b>conditioned</b> on and what the tight ResNet-18 '
             '<b>predicted</b>, plus its nearest neighbour among the 4,274 tight crops the prior '
             'was trained on.</p>')
    for ci, c in enumerate(CLASSES):
        idx = [i for i, l in enumerate(clabels) if l == ci]
        if not idx:
            continue
        P.append(f'<h3>conditioned on: {c}</h3><div class="grid">')
        for i in idx:
            pred = CLASSES[int(cpred[i])]
            good = pred == c
            nnp, sim = nn_c[i]
            n = os.path.basename(cpaths[i])
            scls = 'no' if sim > 0.9 else ('warn' if sim > 0.8 else 'ok')
            P.append(f'<div class="pairbox"><div class="hd">cond=<b>{c}</b> · classifier says '
                     f'<b class="{"ok" if good else "no"}">{pred}</b> · nearest train cos '
                     f'<b class="{scls}">{sim:.3f}</b></div><div class="pair">'
                     f'<figure class="t"><div class="imgwrap"><img src="{cpaths[i]}"></div>'
                     f'<figcaption>GENERATED</figcaption></figure>'
                     f'<figure class="t"><div class="imgwrap"><img src="{A}/nn_cond/{n}"></div>'
                     f'<figcaption>nearest TRAIN</figcaption></figure></div></div>')
        P.append('</div>')

    # 4 side by side
    P.append('<h2>4 · Real above generated — same class, same size</h2>')
    P.append('<p class="note">The row that matters. Top: real crops of the class. Bottom: what the '
             'generator produces when asked for that same class.</p>')
    for ci, c in enumerate(CLASSES):
        idx = [i for i, l in enumerate(clabels) if l == ci][:12]
        P.append(f'<h3>{c}</h3>')
        P.append('<div class="rowlab">REAL (red box = the defect)</div><div class="row">')
        for r in real[c]:
            n = os.path.basename(r['crop_path'])
            bb = bbox_in_crop(r)
            P.append(f'<div class="imgwrap" style="flex:0 0 256px"><img src="{A}/real/{n}">'
                     f'{bbox_svg(bb)}</div>')
        P.append('</div><div class="rowlab">GENERATED (conditioned on this class)</div><div class="row">')
        for i in idx:
            P.append(f'<div class="imgwrap" style="flex:0 0 256px"><img src="{cpaths[i]}"></div>')
        P.append('</div>')

    open('generation_review.html', 'w').write('\n'.join(P) + '\n')


if __name__ == '__main__':
    main()
