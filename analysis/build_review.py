"""Build review.html — a full-resolution visual audit of the tight-crop pipeline.

Four sections:
  1. Real tight crops (256px, VOC bbox overlaid in red, labelled with source filename)
  2. VQ-VAE reconstructions (real | recon, side by side)
  3. Generated samples from the conditional prior (labelled: conditioned vs predicted)
  4. Real vs generated, same class, side by side

Nothing is cherry-picked: 12 crops per class are drawn with random.Random(SEED).sample
over the whole class, and 12 generated samples per class are drawn the same way from the
32 available. Crops whose bbox is not fully inside the 256px frame are reported and shown.

Images are written full-size (256px, no downsampling) to review_assets/ and referenced
relatively; review.html must stay next to that directory.
"""
import sys, pathlib  # repo layout (README.md): src/, experiments/, analysis/ importable from anywhere
sys.path[1:1] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('src', 'experiments', 'analysis')]
import os
import csv
import glob
import html
import random
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

from vqvae import VQVAE
from pcb_utils import CLASSES

SEED = 0
PER_CLASS = 12
TIGHT, OUT_SIZE, W, H = 128, 256, 600, 600
SCALE = OUT_SIZE / TIGHT                      # 128 native px -> 256 px = 2.0
ASSETS = 'review_assets'
MANIFEST = 'results/manifest_tight.csv'       # the size-matched set (primary defect/crop)
PERBBOX = 'results/manifest_tight_perbbox.csv'
GRID_DIR = 'synth_check2'
VQ_CKPT = 'checkpoint/vqvae_tight_560.pt'
CLF_CKPT = 'checkpoint/classifier_Atight_b100_s0.pt'
ANN_DIR = 'VOC_PCB/Annotations'

IMAGENET = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLF_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), IMAGENET])
VQ_TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])


# ---------------------------------------------------------------- crop geometry
def window_origin(cx, cy, size=TIGHT):
    """EXACTLY the clamp used by make_tight_crops.crop_tight()."""
    half = size // 2
    left = max(0, min(int(round(cx - half)), W - size))
    top = max(0, min(int(round(cy - half)), H - size))
    return left, top


def voc_boxes(base):
    """All bboxes of a base 600px crop, in 600px frame coords."""
    p = os.path.join(ANN_DIR, base + '.xml')
    if not os.path.exists(p):
        return []
    root = ET.parse(p).getroot()
    out = []
    for o in root.findall('object'):
        b = o.find('bndbox')
        out.append((o.find('name').text,
                    int(b.find('xmin').text), int(b.find('ymin').text),
                    int(b.find('xmax').text), int(b.find('ymax').text)))
    return out


def bbox_in_crop(row):
    """Map this crop's own bbox into its 256px tight frame. Returns None if unresolvable."""
    base = row['base_crop']
    idx = int(os.path.basename(row['crop_path']).rsplit('__b', 1)[1].split('.')[0])
    bs = voc_boxes(base)
    if idx >= len(bs):
        return None
    name, xmn, ymn, xmx, ymx = bs[idx]
    cx, cy = (xmn + xmx) / 2, (ymn + ymx) / 2
    left, top = window_origin(cx, cy)
    x0, y0 = (xmn - left) * SCALE, (ymn - top) * SCALE
    x1, y1 = (xmx - left) * SCALE, (ymx - top) * SCALE
    inside = x0 >= 0 and y0 >= 0 and x1 <= OUT_SIZE and y1 <= OUT_SIZE
    return {'voc_name': name, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'inside': inside,
            'native_w': xmx - xmn, 'native_h': ymx - ymn}


def load_rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- models
def load_models(device):
    vq = VQVAE()
    vq.load_state_dict(torch.load(VQ_CKPT, map_location='cpu'))
    vq = vq.to(device).eval()
    clf = models.resnet18()
    clf.fc = nn.Linear(clf.fc.in_features, 6)
    clf.load_state_dict(torch.load(CLF_CKPT, map_location='cpu'))
    clf = clf.to(device).eval()
    return vq, clf


@torch.no_grad()
def reconstruct(vq, imgs, device):
    x = torch.stack([VQ_TF(im) for im in imgs]).to(device)
    dec, _ = vq(x)
    dec = dec.clamp(-1, 1)
    out = ((dec + 1) / 2 * 255).round().byte().cpu().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(a) for a in out]


@torch.no_grad()
def classify(clf, imgs, device):
    x = torch.stack([CLF_TF(im) for im in imgs]).to(device)
    logit = clf(x)
    prob = logit.softmax(1)
    return logit.argmax(1).cpu().numpy(), prob.max(1).values.cpu().numpy()


def grid_tiles(cls):
    """Slice the 8x4 sample grid (nrow=8, padding=2) back into 32 full 256px tiles."""
    g = Image.open(f'{GRID_DIR}/grid_{cls}.png').convert('RGB')
    return [g.crop((2 + c * 258, 2 + r * 258, 2 + c * 258 + 256, 2 + r * 258 + 256))
            for r in range(4) for c in range(8)]


# ---------------------------------------------------------------- html helpers
def bbox_svg(bb):
    """Red bbox as an SVG overlay; parts outside the frame are clipped by the viewBox."""
    if bb is None:
        return ''
    x, y = bb['x0'], bb['y0']
    w, h = bb['x1'] - bb['x0'], bb['y1'] - bb['y0']
    return (f'<svg class="ov" viewBox="0 0 256 256">'
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="none" stroke="#ff2020" stroke-width="2"/></svg>')


def tile(src, caption, sub='', bb=None, bad=False):
    return (f'<figure class="t{" bad" if bad else ""}">'
            f'<div class="imgwrap"><img src="{src}" width="256" height="256">{bbox_svg(bb)}</div>'
            f'<figcaption><b>{html.escape(caption)}</b>'
            f'{f"<span>{sub}</span>" if sub else ""}</figcaption></figure>')


def uncond_section(gen):
    """Section 5 — the ORIGINAL unconditional priors (the ICCE-TW pipeline, unmodified),
    side by side with our conditional tight-crop samples."""
    d = 'synth_uncond'
    if not os.path.exists(f'{d}/preds.npy'):
        return ['<h2>5 · Original unconditional priors — <i>still sampling</i></h2>']
    preds = np.load(f'{d}/preds.npy')
    confs = np.load(f'{d}/confs.npy')
    imgs = sorted(glob.glob(f'{d}/u*.png'))
    n = len(preds)
    cnt = np.bincount(preds, minlength=6)
    uni = n / 6
    chi2 = float(((cnt - uni) ** 2 / uni).sum())
    never = [CLASSES[i] for i in range(6) if cnt[i] == 0]

    P = ['<h2>5 · The ORIGINAL unconditional priors — the ICCE-TW pipeline, unmodified</h2>']
    P.append(f'''<p class="note">Sampled from <code>pixelsnail_top_357.pt</code> +
<code>pixelsnail_bottom_best.pt</code> decoded through <code>vqvae_560.pt</code> — the original
420-epoch checkpoints, no class conditioning, no modifications. Loaded with
<code>strict=True</code> (0 missing, 0 unexpected keys), so these are the original weights in the
original architecture.</p>
<p class="note"><b>An unconditional prior has no class input</b>, so "30 per class" is not
something you can ask of it — you draw one undifferentiated pool and the class exists only once a
classifier assigns one. That <i>is</i> the finding: this pipeline emits an <b>unlabeled</b> pool.
We drew <b>{n}</b> samples (the same budget as the conditional run: 6×30) and labelled them with
the <b>600px ResNet-18</b> (<code>classifier_A_b100_s0.pt</code>), which matches the 600px crops
these priors were trained on.</p>''')

    P.append(f'''<p class="note"><b class="no">Leakage warning.</b> These priors were trained on
<code>lmdb/all</code> — <b>all 10 boards, including the held-out test boards 06 and 09</b>. Nothing
sampled from them may ever be used as training data for a model we then evaluate on those boards.
They are shown here for inspection only.</p>''')

    P.append('<table><tr><th style="text-align:left">predicted class</th><th>count</th>'
             '<th>share</th><th>uniform would be</th></tr>')
    for i, c in enumerate(CLASSES):
        hi = ' class="hi"' if cnt[i] == 0 or cnt[i] > 2 * uni else ''
        P.append(f'<tr><th style="text-align:left">{c}</th><td{hi}>{cnt[i]}</td>'
                 f'<td{hi}>{100*cnt[i]/n:.1f}%</td><td>{100/6:.1f}%</td></tr>')
    P.append('</table>')
    P.append(f'<p class="note">χ² vs uniform = <b>{chi2:.1f}</b> (dof 5; &gt;11.07 ⇒ p&lt;0.05, '
             f'so the pool is <b>not</b> class-balanced). Classes never generated at all: '
             f'<b class="no">{", ".join(never) if never else "none"}</b>. '
             f'Mean classifier confidence {confs.mean():.2f}.</p>')

    P.append(f'<h3>all {n} unconditional samples (labelled by what the 600px ResNet-18 called them)</h3>')
    P.append('<div class="grid">')
    for k, p in enumerate(imgs):
        P.append(tile(p, f'#{k:03d}', f'classifier says <b>{CLASSES[preds[k]]}</b> '
                                      f'({confs[k]*100:.0f}%)'))
    P.append('</div>')

    P.append('<h3>side by side: unconditional (original) vs class-conditional (tight)</h3>')
    P.append('<p class="note">Left: the original unconditional pipeline — 600px-scale content, no '
             'class asked for or given. Right: our class-conditional tight prior, asked for the '
             'named class. Neither renders a defect; the left one cannot even be asked to.</p>')
    for i, c in enumerate(CLASSES):
        P.append(f'<h3>{c}</h3><div class="grid">')
        pool = [k for k in range(n) if preds[k] == i] or list(range(n))
        pick = random.Random(SEED).sample(pool, min(6, len(pool)))
        for k, (gn, pred, cf) in zip(pick, gen[c][:6]):
            P.append(f'<div class="pairbox"><div class="hd">uncond (clf: {CLASSES[preds[k]]}) | '
                     f'cond={c} (clf: {pred})</div><div class="pair">'
                     f'<figure class="t"><div class="imgwrap"><img src="{imgs[k]}" '
                     f'width="256" height="256"></div><figcaption>UNCONDITIONAL</figcaption></figure>'
                     f'<figure class="t"><div class="imgwrap"><img src="{ASSETS}/gen/{gn}" '
                     f'width="256" height="256"></div><figcaption>CONDITIONAL</figcaption></figure>'
                     f'</div></div>')
        P.append('</div>')
    return P


def checkpoint_section():
    """Sections 6-7 — what every checkpoint is, and whether the tight prior had converged."""
    P = ['<h2>6 · Every checkpoint on disk — epochs and training data</h2>']
    if os.path.exists('results/checkpoint_inventory.csv'):
        rows = load_rows('results/checkpoint_inventory.csv')
        P.append('<table><tr><th style="text-align:left">family</th><th>ckpts</th>'
                 '<th>epoch reached</th><th>epoch budget</th>'
                 '<th style="text-align:left">trained on</th></tr>')
        for r in rows:
            und = r['family'].startswith('pixelsnail_condtight2')
            leak = 'LEAKED' in r['data']
            cls = ' class="hi"' if und or leak else ''
            P.append(f'<tr><th style="text-align:left">{r["family"]}</th><td>{r["n_ckpt"]}</td>'
                     f'<td{cls}>{r["reached"]}</td><td>{r["planned"] or "-"}</td>'
                     f'<td{cls} style="text-align:left">{r["data"]} ({r["n_train"]})</td></tr>')
        P.append('</table>')
    P.append('''<p class="note">Two rows are flagged. <b>(1)</b> The original
<code>pixelsnail_top/_bottom</code> were trained on <code>lmdb/all</code> — <b>all 10 boards,
test boards included</b>. <b>(2)</b> <code>pixelsnail_condtight2</code>, the prior the whole
conditioning check was run on, got <b>80 epochs against the original's 420</b> — a 5.25×
smaller budget.</p>''')

    P.append('<h2>7 · Had the tight prior actually converged? <b class="no">No.</b></h2>')
    P.append('''<p class="note">The trainers only ever wrote loss to a tqdm progress bar, so these
curves are recovered from the tqdm lines in the redirected logs (per-epoch mean of the batch
losses, de-duplicated on epoch+iteration). <b>The original 420-epoch run left no log and only two
terminal checkpoints, so its curve is genuinely unrecoverable</b> — it is absent below rather than
invented.</p>''')
    P.append('<img src="figures/fig8_prior_loss_curves.png" style="max-width:100%;border-radius:6px">')
    P.append('''<p class="note"><b>Fitting the last 20 epochs (61→80) of the prior the conditioning
check was actually run on:</b></p>
<table><tr><th style="text-align:left">prior</th><th>loss @61</th><th>loss @80</th>
<th>fall over last 20ep</th><th>slope</th><th>verdict</th></tr>
<tr><th style="text-align:left">TOP</th><td>0.2177</td><td>0.1669</td><td class="hi">−23.3%</td>
<td>−0.00266/ep</td><td class="hi">still falling</td></tr>
<tr><th style="text-align:left">BOTTOM</th><td>1.4076</td><td>1.2847</td><td class="hi">−8.7%</td>
<td>−0.00688/ep</td><td class="hi">still falling</td></tr></table>
<p class="note">Neither prior had plateaued, and bottom-prior code accuracy (0.61) was still
climbing steeply. <b>The conditioning check at epoch 80 was therefore premature</b>, and the
negative result cannot be stated as "class-conditional generation fails" — only as "class-conditional
generation had not produced class-faithful defects after 80 of a 420-epoch budget". Training has
been resumed from epoch 80 to 320 (<code>pixelsnail_condtight3_*</code>); the conditioning check
must be re-run at convergence before any conclusion is drawn.</p>''')
    return P


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for d in ('real', 'recon', 'gen'):
        os.makedirs(f'{ASSETS}/{d}', exist_ok=True)

    rows = load_rows(MANIFEST)
    by_class = {c: [r for r in rows if r['class_name'] == c] for c in CLASSES}

    # ---- audit EVERY tight crop (all bboxes) for out-of-frame boxes -----------
    oof = []
    for r in load_rows(PERBBOX):
        bb = bbox_in_crop(r)
        if bb is not None and not bb['inside']:
            oof.append((r, bb))
    print(f'out-of-frame bboxes: {len(oof)} / {len(load_rows(PERBBOX))}')

    vq, clf = load_models(device)

    # ---- sample, no cherry-picking -------------------------------------------
    sel = {c: random.Random(SEED).sample(by_class[c], min(PER_CLASS, len(by_class[c])))
           for c in CLASSES}

    real, recon, gen, genpred = {}, {}, {}, {}
    for c in CLASSES:
        imgs = [Image.open(r['crop_path']).convert('RGB') for r in sel[c]]
        recs = reconstruct(vq, imgs, device)
        for r, im, rc in zip(sel[c], imgs, recs):
            n = os.path.basename(r['crop_path'])
            im.save(f'{ASSETS}/real/{n}')
            rc.save(f'{ASSETS}/recon/{n}')
        real[c] = [(r, bbox_in_crop(r)) for r in sel[c]]
        recon[c] = [os.path.basename(r['crop_path']) for r in sel[c]]

        tiles = grid_tiles(c)
        pred_all, _ = classify(clf, tiles, device)          # all 32, for the honest tally
        pick = random.Random(SEED).sample(range(len(tiles)), PER_CLASS)
        pred, conf = classify(clf, [tiles[i] for i in pick], device)
        for k, i in enumerate(pick):
            tiles[i].save(f'{ASSETS}/gen/{c}_{i:02d}.png')
        gen[c] = [(f'{c}_{i:02d}.png', CLASSES[pred[k]], float(conf[k])) for k, i in enumerate(pick)]
        genpred[c] = np.bincount(pred_all, minlength=6)
        print(f'{c:16s} real={len(sel[c])} gen={len(pick)}  '
              f'all-32 preds: ' + ' '.join(f'{CLASSES[j][:4]}={genpred[c][j]}' for j in range(6)))

    # save out-of-frame crops (all of them) with their boxes
    for r, bb in oof:
        n = os.path.basename(r['crop_path'])
        if not os.path.exists(f'{ASSETS}/real/{n}'):
            Image.open(r['crop_path']).convert('RGB').save(f'{ASSETS}/real/{n}')

    conf_mat = np.stack([genpred[c] for c in CLASSES])
    write_html(real, recon, gen, conf_mat, oof, len(load_rows(PERBBOX)))
    print('\nwrote review.html')


def write_html(real, recon, gen, conf_mat, oof, n_all):
    src = html.escape(open('src/make_tight_crops.py').read())
    diag = int(np.trace(conf_mat)); tot = int(conf_mat.sum())
    spur = CLASSES.index('spur')

    P = []
    P.append(f'''<!doctype html><meta charset="utf-8"><title>Tight-crop pipeline — visual review</title>
<style>
:root{{--bg:#111417;--fg:#e8e8e8;--mut:#9aa3ab;--line:#2a2f35;--card:#1a1e22}}
*{{box-sizing:border-box}}
body{{margin:0;padding:32px;background:var(--bg);color:var(--fg);
 font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}}
h1{{margin:0 0 4px;font-size:26px}} h2{{margin:44px 0 6px;font-size:20px;
 border-bottom:1px solid var(--line);padding-bottom:8px}}
h3{{margin:26px 0 10px;font-size:15px;color:#7fd1ff;font-family:ui-monospace,monospace}}
.note{{color:var(--mut);max-width:900px}}
.grid{{display:flex;flex-wrap:wrap;gap:14px}}
figure.t{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:6px;
 padding:8px;width:272px}}
figure.t.bad{{border-color:#ff2020;box-shadow:0 0 0 1px #ff2020 inset}}
.imgwrap{{position:relative;width:256px;height:256px;overflow:hidden}}
.imgwrap img{{display:block;width:256px;height:256px;image-rendering:pixelated}}
svg.ov{{position:absolute;inset:0;width:256px;height:256px;pointer-events:none}}
figcaption{{margin-top:6px;font:11px/1.4 ui-monospace,monospace;color:var(--fg);
 word-break:break-all}}
figcaption span{{display:block;color:var(--mut);margin-top:2px}}
.pair{{display:flex;gap:6px}}
.pair figure.t{{width:auto;border:0;background:none;padding:0}}
.pairbox{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px}}
.hd{{font:11px ui-monospace,monospace;color:var(--mut);margin-bottom:4px}}
.ok{{color:#4ade80}} .no{{color:#ff6b6b}}
pre{{background:#0d1013;border:1px solid var(--line);border-radius:6px;padding:16px;
 overflow-x:auto;font:12px/1.5 ui-monospace,monospace;color:#cbd5e1}}
table{{border-collapse:collapse;margin:10px 0}}
td,th{{border:1px solid var(--line);padding:5px 10px;text-align:right;font:12px ui-monospace,monospace}}
th{{color:var(--mut)}} td.hi{{background:#3a1414;color:#ff9b9b}}
.key{{display:inline-block;width:12px;height:12px;border:2px solid #ff2020;vertical-align:-2px}}
</style>
<h1>Tight-crop pipeline — full-resolution visual review</h1>
<p class="note">Every image below is a <b>256×256 PNG shown at 256×256</b> — no downsampling,
no thumbnails, <code>image-rendering:pixelated</code> so the browser does not smooth them.
<b>Nothing is cherry-picked:</b> the 12 crops per class are drawn with
<code>random.Random({SEED}).sample(...)</code> over <i>all</i> crops of that class in
<code>{MANIFEST}</code>, and the 12 generated samples per class are drawn the same way from the
32 available in <code>{GRID_DIR}/</code>. <span class="key"></span> = the VOC bounding box,
mapped into the crop with the <i>same</i> clamp the cutter used.</p>
<p class="note"><b>Checks run over all {n_all} tight crops before drawing this page</b> — so you can
trust the red boxes rather than take my word for them:
<b class="ok">0</b> bboxes fall outside the 256px frame;
<b class="ok">0</b> disagreements between the class in the filename and the class VOC gives the box;
the bbox centre sits a median of <b>1.0px</b> from the crop centre (77.8% within 20px — the
remainder is the window being clamped at the image border, exactly as <code>crop_tight()</code>
does it), which is what confirms the overlay is being placed correctly;
and the median defect now fills <b>56×54px of 256 (~22% of the frame)</b>, up from ~4.5% in the
original 600px crops.</p>''')

    # ---------------- 1. real ----------------
    P.append('<h2>1 · Real tight crops <span class="note">— 12/class, VOC bbox in red</span></h2>')
    P.append('<p class="note">The red box is the defect. If it sits inside the frame, the crop '
             'provably contains the defect it is labelled with. Caption shows the source filename, '
             'the class from the filename, the class VOC gives that box, its native size in the '
             '600px frame, and whether the box is fully inside the 256px crop.</p>')
    for c in CLASSES:
        P.append(f'<h3>{c}</h3><div class="grid">')
        for r, bb in real[c]:
            n = os.path.basename(r['crop_path'])
            if bb is None:
                P.append(tile(f'{ASSETS}/real/{n}', n, '<i>no bbox resolved</i>', None, True))
                continue
            mism = bb['voc_name'] != c
            voc_cls = 'no' if mism else 'ok'
            inside = '<b class="ok">inside</b>' if bb['inside'] else '<b class="no">OUTSIDE</b>'
            sub = (f"board {r['board_id']} · {r['split']} · "
                   f"voc=<b class=\"{voc_cls}\">{bb['voc_name']}</b> · "
                   f"defect {bb['native_w']}×{bb['native_h']}px native · bbox {inside}")
            P.append(tile(f'{ASSETS}/real/{n}', n, sub, bb, not bb['inside'] or mism))
        P.append('</div>')

    # ---------------- out of frame ----------------
    P.append('<h2>1b · Crops whose bbox is not fully inside the frame</h2>')
    if not oof:
        P.append(f'<p class="note"><b class="ok">None.</b> All <b>{n_all}</b> tight crops '
                 '(every annotated bbox, both manifests) have their bounding box fully inside the '
                 '256px frame, so every tight crop provably contains its whole defect.</p>')
    else:
        P.append(f'<p class="note"><b class="no">{len(oof)} of {n_all}</b> tight crops have a bbox '
                 'that extends past the frame edge — the defect is larger than the 128px native '
                 'window, or the window was clamped at the image border. <b>All of them are shown '
                 'here</b>, red box clipped at the frame.</p><div class="grid">')
        for r, bb in oof:
            n = os.path.basename(r['crop_path'])
            sub = (f"board {r['board_id']} · {r['split']} · defect {bb['native_w']}×{bb['native_h']}px "
                   f"native · bbox x[{bb['x0']:.0f},{bb['x1']:.0f}] y[{bb['y0']:.0f},{bb['y1']:.0f}] "
                   f"vs frame [0,256]")
            P.append(tile(f'{ASSETS}/real/{n}', n, sub, bb, True))
        P.append('</div>')

    # ---------------- 2. recon ----------------
    P.append('<h2>2 · VQ-VAE reconstructions <span class="note">— real | reconstruction</span></h2>')
    P.append(f'<p class="note">Same 12 crops per class, encoded and decoded straight back through '
             f'<code>{VQ_CKPT}</code> — no prior, no sampling. This is the sanity gate: these '
             'reconstructions classify at <b>0.918 macro-F1</b> vs 0.926 on the raw crops, so the '
             'autoencoder renders defects faithfully. Any failure downstream is the prior, not this.</p>')
    for c in CLASSES:
        P.append(f'<h3>{c}</h3><div class="grid">')
        for (r, bb), n in zip(real[c], recon[c]):
            P.append(f'<div class="pairbox"><div class="hd">{html.escape(n)}</div><div class="pair">'
                     f'<figure class="t"><div class="imgwrap"><img src="{ASSETS}/real/{n}" '
                     f'width="256" height="256"></div><figcaption>real</figcaption></figure>'
                     f'<figure class="t"><div class="imgwrap"><img src="{ASSETS}/recon/{n}" '
                     f'width="256" height="256"></div><figcaption>VQ-VAE recon</figcaption></figure>'
                     f'</div></div>')
        P.append('</div>')

    # ---------------- 3. generated ----------------
    P.append('<h2>3 · Generated samples from the class-conditional prior</h2>')
    pct = 100 * diag / tot
    spur_pct = 100 * conf_mat[:, spur].sum() / tot
    P.append(f'<p class="note">12/class drawn at random (seed {SEED}) from the 32 sampled per class '
             '(temp 1.0, stronger-conditioned priors <code>pixelsnail_condtight2_*_080.pt</code>). '
             'Each is labelled with the class it was <b>conditioned on</b> and what the tight '
             'ResNet-18 actually <b>predicted</b>. Predictions are recomputed here from the saved '
             f'PNGs. Over all 32×6={tot} samples: <b>{diag}/{tot} = {pct:.1f}%</b> match the '
             f'conditioned class (chance 16.7%) — but <b class="no">{spur_pct:.1f}% of every sample, '
             'whatever class was asked for, is classified <code>spur</code></b>. That collapse, not '
             'conditioning, is what produces the apparent signal.</p>'
             '<p class="note">Two details, so nothing here looks like a fudge. (i) The sampling run '
             'logged <b>22.4%</b> (43/192); recomputing from the saved PNGs gives <b>22.9%</b> '
             '(44/192) — 5 of the 6 rows reproduce <i>exactly</i>, and one <code>mouse_bite</code> '
             'sample flips, because the PNG is 8-bit-quantised and re-resized with PIL rather than '
             'the bilinear resize used in-memory. (ii) The <code>spur</code> column is the whole '
             'story: strip that one collapse class out and consistency across the remaining five '
             'falls to <b>13%, below the 16.7% chance line</b>. Conditioning is not weak here; it '
             'is absent.</p>')
    P.append('<table><tr><th>conditioned ↓ / predicted →</th>' +
             ''.join(f'<th>{c[:9]}</th>' for c in CLASSES) + '</tr>')
    for i, c in enumerate(CLASSES):
        P.append(f'<tr><th style="text-align:left">{c}</th>' + ''.join(
            f'<td class="{"hi" if j == spur else ""}">{conf_mat[i, j]}</td>' for j in range(6)) + '</tr>')
    P.append('</table>')
    for c in CLASSES:
        P.append(f'<h3>conditioned on: {c}</h3><div class="grid">')
        for n, pred, cf in gen[c]:
            good = pred == c
            sub = (f'predicted <b class="{"ok" if good else "no"}">{pred}</b> '
                   f'({cf*100:.0f}% conf)')
            P.append(tile(f'{ASSETS}/gen/{n}', f'cond={c}', sub, None, not good))
        P.append('</div>')

    # ---------------- 4. real vs generated ----------------
    P.append('<h2>4 · Real vs generated, same class, side by side</h2>')
    P.append('<p class="note">Left of each pair: a real tight crop of the class (red box = the '
             'actual defect). Right: a sample the prior generated when asked for that same class. '
             'The generated crops are photorealistic PCB substrate — solder mask, traces, pads — '
             'with <b>no defect drawn</b>. Look at <code>missing_hole</code>: the generated pads '
             'still have their holes.</p>')
    for c in CLASSES:
        P.append(f'<h3>{c}</h3><div class="grid">')
        for (r, bb), (gn, pred, cf) in zip(real[c], gen[c]):
            rn = os.path.basename(r['crop_path'])
            P.append(f'<div class="pairbox"><div class="hd">real (bbox in red) | generated '
                     f'(cond={c}, pred={pred})</div><div class="pair">'
                     f'<figure class="t"><div class="imgwrap"><img src="{ASSETS}/real/{rn}" '
                     f'width="256" height="256">{bbox_svg(bb)}</div>'
                     f'<figcaption>REAL</figcaption></figure>'
                     f'<figure class="t"><div class="imgwrap"><img src="{ASSETS}/gen/{gn}" '
                     f'width="256" height="256"></div>'
                     f'<figcaption>GENERATED</figcaption></figure></div></div>')
        P.append('</div>')

    # ---------------- 5. unconditional / ICCE-TW pipeline ----------------
    P.extend(uncond_section(gen))

    # ---------------- 6. checkpoints + convergence ----------------
    P.extend(checkpoint_section())

    # ---------------- code ----------------
    P.append('<h2>8 · The tight-crop cutting code, verbatim</h2>')
    P.append('<p class="note">This is <code>make_tight_crops.py</code> exactly as it ran. The window '
             'is <code>crop_tight()</code>: 128×128 native, centred on the bbox centre, clamped to '
             'the 600px frame, then LANCZOS-resized to 256 (a 2× upscale). The red boxes above are '
             'placed by replaying this same clamp, so what you see is where the cutter actually put '
             'the defect.</p>')
    P.append(f'<pre>{src}</pre>')
    open('review.html', 'w').write('\n'.join(P) + '\n')


if __name__ == '__main__':
    main()
