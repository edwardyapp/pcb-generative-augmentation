"""duplicates_review.html — the visual evidence, including the evidence that I WAS WRONG.

THE RETRACTED CLAIM. I reported that the 600px crop set duplicates the same photograph under
several class labels, that 90.4% of it is affected, and that the task therefore has a 27.2%
accuracy ceiling that our baseline had already reached. That is FALSE. Every part of it.

WHY IT WAS FALSE.
  * I clustered images at 64x64. A 27px defect in a 600px frame is invisible at 64x64, so I was
    clustering BACKGROUND TEMPLATES and calling them duplicate IMAGES.
  * I then "confirmed" duplication with mean-absolute-error over the full frame. But the defect
    is ~0.07% of the pixels: two images differing ONLY in the defect would still show an MAE of
    ~0.1/255. MAE literally cannot see the thing in question. My own output showed max differences
    of 64-189/255 -- huge, localised -- and I read past it.

WHAT IS ACTUALLY TRUE. The images are NOT byte-identical, and their differences land exactly on
the union of the two images' bounding boxes. They are the SAME PCB TEMPLATE with DIFFERENT
DEFECTS INJECTED. Each image really does contain the defect it is labelled with. The labels are
sound, there is no ceiling, and the 600px baseline is low for the reason we always thought: a
27px defect in a 600px frame is a needle, and after the 224 resize it is ~10px.

This page shows, for each pair: the two crops at full 600x600 with their OWN bbox in red, and the
amplified difference map with both boxes drawn on it. The difference lights up in the boxes and
nowhere else. That is the whole argument, and you can check it with your eyes.
"""
import hashlib
import html
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

IMG, ANN = 'VOC_PCB/JPEGImages', 'VOC_PCB/Annotations'
OUT = 'dupreview_assets'
AMP = 6                     # amplify the difference so it is visible at all

PAIRS = [
    ('light_01_mouse_bite_01_1_600', 'light_01_short_02_1_600'),
    ('light_01_mouse_bite_01_1_600', 'light_01_spur_05_1_600'),
    ('light_01_open_circuit_04_2_600', 'light_01_spurious_copper_04_1_600'),
    ('light_01_mouse_bite_01_2_600', 'light_01_spurious_copper_01_2_600'),
    ('light_01_mouse_bite_02_1_600', 'light_01_open_circuit_02_1_600'),
    ('light_01_short_01_2_600', 'light_01_spur_02_1_600'),
]


def boxes(b):
    r = ET.parse(f'{ANN}/{b}.xml').getroot()
    return [(o.find('name').text,) + tuple(int(o.find('bndbox').find(k).text)
                                           for k in ('xmin', 'ymin', 'xmax', 'ymax'))
            for o in r.findall('object')]


def svg(bs, color):
    s = ''
    for name, x0, y0, x1, y1 in bs:
        s += (f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" '
              f'stroke="{color}" stroke-width="4"/>'
              f'<text x="{x0}" y="{max(y0-6,14)}" fill="{color}" font-size="20" '
              f'font-family="monospace">{name}</text>')
    return f'<svg class="ov" viewBox="0 0 600 600">{s}</svg>'


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for a, b in PAIRS:
        fa, fb = f'{IMG}/{a}.jpg', f'{IMG}/{b}.jpg'
        A = np.asarray(Image.open(fa).convert('RGB'), float)
        B = np.asarray(Image.open(fb).convert('RGB'), float)
        md5a = hashlib.md5(open(fa, 'rb').read()).hexdigest()
        md5b = hashlib.md5(open(fb, 'rb').read()).hexdigest()

        D = np.abs(A - B)
        Dm = D.max(2)
        diff = np.clip(D * AMP, 0, 255).astype(np.uint8)
        Image.open(fa).convert('RGB').save(f'{OUT}/{a}.png')
        Image.open(fb).convert('RGB').save(f'{OUT}/{b}.png')
        Image.fromarray(diff).save(f'{OUT}/diff_{a}__{b}.png')

        ba, bb = boxes(a), boxes(b)
        # what fraction of the strongly-differing pixels falls INSIDE one of the two boxes?
        mask = np.zeros(Dm.shape, bool)
        strict = np.zeros(Dm.shape, bool)
        for _, x0, y0, x1, y1 in ba + bb:
            mask[max(0, y0 - 8):y1 + 8, max(0, x0 - 8):x1 + 8] = True
            strict[y0:y1, x0:x1] = True
        strong = Dm > 30
        inside = float((strong & mask).sum() / max(1, strong.sum()))
        Dmean = D.mean(2)
        rows.append(dict(a=a, b=b, md5a=md5a, md5b=md5b, same=md5a == md5b,
                         mae=float(D.mean()), mx=float(Dm.max()),
                         mae_in=float(Dmean[strict].mean()),
                         mae_out=float(Dmean[~strict].mean()),
                         area=float(100 * strict.mean()),
                         nstrong=int(strong.sum()), inside=inside, ba=ba, bb=bb))
        print(f'{a} vs {b}: md5same={md5a==md5b} MAE={D.mean():.3f} MAX={Dm.max():.0f} '
              f'{100*inside:.1f}% of differing pixels are inside a bbox')

    mean_inside = np.mean([r['inside'] for r in rows])
    P = [f'''<!doctype html><meta charset="utf-8"><title>The "duplicate" pairs — a retraction</title>
<style>
body{{margin:0;padding:30px;background:#111417;color:#e8e8e8;font:14px/1.6 -apple-system,Segoe UI,Roboto,sans-serif}}
h1{{margin:0 0 8px}} h2{{margin:38px 0 8px;border-bottom:1px solid #2a2f35;padding-bottom:8px}}
.note{{color:#9aa3ab;max-width:1000px}}
.bad{{background:#3a1414;border:1px solid #ff6b6b;border-radius:8px;padding:16px 20px;max-width:1000px}}
.good{{background:#12331e;border:1px solid #4ade80;border-radius:8px;padding:16px 20px;max-width:1000px}}
.row{{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0 6px}}
figure{{margin:0;background:#1a1e22;border:1px solid #2a2f35;border-radius:6px;padding:8px}}
.wrap{{position:relative;width:600px;height:600px}}
.wrap img{{display:block;width:600px;height:600px;image-rendering:pixelated}}
svg.ov{{position:absolute;inset:0;width:600px;height:600px}}
figcaption{{margin-top:6px;font:12px ui-monospace,monospace;color:#cbd5e1;word-break:break-all}}
code{{color:#7fd1ff}} .no{{color:#ff6b6b}} .ok{{color:#4ade80}}
table{{border-collapse:collapse;margin:12px 0}}
td,th{{border:1px solid #2a2f35;padding:6px 12px;font:12px ui-monospace,monospace;text-align:right}}
th{{color:#9aa3ab;text-align:left}}
</style>
<h1>The "duplicate image" pairs — see for yourself</h1>
<div class="bad">
<b>RETRACTION.</b> I claimed the 600px crop set files <b>the same photograph under several
different class labels</b>, that 90.4% of it is affected, and that this imposes a <b>27.2% accuracy
ceiling</b> which our baseline had already hit. <b>That was wrong — all of it.</b>
<ul>
<li>I clustered the images at <b>64×64</b>. A 27px defect in a 600px frame <i>does not exist</i> at
64×64. I was clustering <b>background templates</b> and calling them duplicate <b>images</b>.</li>
<li>I "confirmed" it with mean-absolute-error over the whole frame. The defect is ~0.07% of the
pixels, so two images differing <i>only</i> in the defect still show an MAE around 0.1/255.
<b>MAE cannot see the one thing being argued about.</b> My own output showed max differences of
64–189/255 — enormous and localised — and I read straight past them.</li>
</ul>
</div>
<div class="good" style="margin-top:14px">
<b>WHAT IS ACTUALLY TRUE.</b> The images are <b>not byte-identical</b>, and their differences land
<b>exactly on the two images' bounding boxes</b> — across these 6 pairs,
<b>{100*mean_inside:.0f}% of all strongly-differing pixels fall inside a box</b>.
These are the <b>same PCB template with different defects injected</b>. Each image really does
contain the defect it is labelled with. <b>The labels are sound. There is no ceiling.</b>
The 600px baseline is low for the reason we always thought: a 27px defect in a 600px frame is a
needle, and after the 224 resize it is about 10px across.
</div>
<h2>Why the metric was broken: MAE literally cannot see a defect</h2>
<p class="note">A 27px defect occupies ~0.27% of a 600×600 frame. So even when a defect is
<b>completely different</b> between two images, it can only move the <i>global</i> mean-absolute-error
by a hair. Measured on these very pairs:</p>
<table>
<tr><th>quantity</th><th>MAE (/255)</th><th>what it means</th></tr>
<tr><th>INSIDE the bounding boxes</th><td class="ok">5.3 – 12.9</td>
    <td style="text-align:left">the defect is really there, and it is <b>large</b></td></tr>
<tr><th>OUTSIDE the boxes (the template)</th><td>0.005 – 0.009</td>
    <td style="text-align:left">the backgrounds are <b>digitally identical</b></td></tr>
<tr><th>GLOBAL — the number I used</th><td class="no">0.02 – 0.04</td>
    <td style="text-align:left">the defect is diluted 1:370 by the background</td></tr>
<tr><th>pure JPEG re-encode noise</th><td class="no">0.745</td>
    <td style="text-align:left"><b>25× bigger than the defect's global signal</b></td></tr>
<tr><th>two genuinely different boards</th><td>40.3 ± 6.1</td>
    <td style="text-align:left">this is what a real "different image" looks like</td></tr>
</table>
<p class="note">My "identical image" threshold was <b>MAE &lt; 1.0</b>. That is <i>above the JPEG
noise floor</i> and <b>~30× above the entire signal a real defect produces</b>. It could not
possibly distinguish "the same photograph" from "the same board with a different defect injected".
The duplicate detection was not merely noisy — it was <b>structurally incapable</b> of measuring
the thing it was being used to rule out, and it returned exactly the answer that blindness
guarantees.</p>
<p class="note" style="margin-top:16px">Each pair below: the two crops at <b>full 600×600</b>, each
with <b>its own</b> bounding box in red, then the <b>contrast-stretched difference map</b>
(amplified ×{AMP}) with <b>both</b> boxes drawn on it — <span style="color:#ff2020">red = image A's
defect</span>, <span style="color:#00d0ff">cyan = image B's defect</span>. The difference lights up
inside the boxes and is black everywhere else. That is the whole argument.</p>''']

    for r in rows:
        P.append(f'<h2>{html.escape(r["a"])} &nbsp;vs&nbsp; {html.escape(r["b"])}</h2>')
        P.append(f'''<table>
<tr><th>byte-identical (md5)?</th><td class="{'no' if r['same'] else 'ok'}">
    {'YES' if r['same'] else 'NO — different files'}</td></tr>
<tr><th>mean abs difference</th><td>{r['mae']:.3f}/255 &nbsp;<span style="color:#9aa3ab">
    (looks like "identical" — but this is the misleading number)</span></td></tr>
<tr><th>MAX abs difference</th><td class="no">{r['mx']:.0f}/255 &nbsp;<span style="color:#9aa3ab">
    (huge — and localised)</span></td></tr>
<tr><th>MAE <b>inside</b> the boxes</th><td class="ok">{r['mae_in']:.2f}/255 &nbsp;<span style="color:#9aa3ab">
    (the defect — real and large)</span></td></tr>
<tr><th>MAE <b>outside</b> the boxes</th><td>{r['mae_out']:.3f}/255 &nbsp;<span style="color:#9aa3ab">
    (the shared template — digitally identical)</span></td></tr>
<tr><th>boxes as % of frame</th><td>{r['area']:.2f}%</td></tr>
<tr><th>pixels differing &gt;30/255</th><td>{r['nstrong']}</td></tr>
<tr><th>…of those, inside a bbox</th><td class="ok">{100*r['inside']:.1f}%</td></tr>
</table>''')
        P.append('<div class="row">')
        P.append(f'<figure><div class="wrap"><img src="{OUT}/{r["a"]}.png">{svg(r["ba"], "#ff2020")}</div>'
                 f'<figcaption>A: {html.escape(r["a"])}.jpg<br>md5 {r["md5a"][:16]}…</figcaption></figure>')
        P.append(f'<figure><div class="wrap"><img src="{OUT}/{r["b"]}.png">{svg(r["bb"], "#00d0ff")}</div>'
                 f'<figcaption>B: {html.escape(r["b"])}.jpg<br>md5 {r["md5b"][:16]}…</figcaption></figure>')
        P.append(f'<figure><div class="wrap"><img src="{OUT}/diff_{r["a"]}__{r["b"]}.png">'
                 f'{svg(r["ba"], "#ff2020")}{svg(r["bb"], "#00d0ff")}</div>'
                 f'<figcaption>|A − B| × {AMP}. Bright only inside the boxes ⇒ same template,<br>'
                 f'different injected defects. NOT the same image.</figcaption></figure>')
        P.append('</div>')

    open('duplicates_review.html', 'w').write('\n'.join(P) + '\n')
    print(f'\nmean: {100*mean_inside:.1f}% of differing pixels fall inside a bbox')
    print('wrote duplicates_review.html')


if __name__ == '__main__':
    main()
