"""fig_samples.png — real vs b100-generated defect crops, 2x3, for the paper.

Top row: real defect-centred tight crops (PCB-cropped-tight/all, 256x256,
un-augmented originals only: no rotation_/l_ variants). Bottom row: b100
conditional generator samples (synth_b100), same classes.

NO CHERRY-PICKING: one random.Random(0) instance; picks drawn with
rng.choice(sorted(pool)) in the fixed order real:[missing_hole, short, spur]
then generated:[missing_hole, short, spur]. Chosen files recorded in
fig_samples_NOTE.txt.

B/W-print safe: black text on white, no colour-coded annotations.
Output: fig_samples.png (300 dpi) + fig_samples_NOTE.txt in project root.
"""
import os
import random
import re

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["missing_hole", "short", "spur"]
TILE, GAP = 256, 4
GUTTER_W, HEADER_H, MARGIN = 92, 30, 6
SEED = 0

# strict anchored pattern: original tight crops only (no rotation_/l_ prefix);
# full-class alternation avoids the spur/spurious_copper prefix collision.
_REAL_PAT = re.compile(
    r"^light_\d+_(missing_hole|short|spur)_\d+_\d+_600(?:__b\d+)?\.png$")


def real_pool(cls):
    d = os.path.join(ROOT, "PCB-cropped-tight", "all")
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if (m := _REAL_PAT.match(f)) and m.group(1) == cls)


def gen_pool(cls):
    d = os.path.join(ROOT, "synth_b100")
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if re.match(rf"^{cls}_\d+\.png$", f))


def load_font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    rng = random.Random(SEED)
    picks = {}   # (row, cls) -> path, drawn in documented fixed order
    for row, pool_fn in (("real", real_pool), ("generated", gen_pool)):
        for cls in CLASSES:
            pool = pool_fn(cls)
            if not pool:
                raise SystemExit(f"empty pool: {row}/{cls}")
            picks[(row, cls)] = rng.choice(pool)

    W = MARGIN + GUTTER_W + 3 * TILE + 2 * GAP + MARGIN
    H = MARGIN + HEADER_H + 2 * TILE + GAP + MARGIN
    fig = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(fig)
    f_col, f_row = load_font(17), load_font(16)

    x0, y0 = MARGIN + GUTTER_W, MARGIN + HEADER_H
    for j, cls in enumerate(CLASSES):
        cx = x0 + j * (TILE + GAP)
        w = draw.textlength(cls, font=f_col)
        draw.text((cx + (TILE - w) / 2, MARGIN + 4), cls, fill="black", font=f_col)
    for i, row in enumerate(("real", "generated")):
        ry = y0 + i * (TILE + GAP)
        bb = draw.textbbox((0, 0), row, font=f_row)
        draw.text((MARGIN + (GUTTER_W - 8 - (bb[2] - bb[0])) ,
                   ry + (TILE - (bb[3] - bb[1])) / 2), row,
                  fill="black", font=f_row)
        for j, cls in enumerate(CLASSES):
            im = Image.open(picks[(row, cls)]).convert("RGB")
            if im.size != (TILE, TILE):
                raise SystemExit(f"{picks[(row, cls)]} is {im.size}, not 256x256")
            fig.paste(im, (x0 + j * (TILE + GAP), ry))

    out = os.path.join(ROOT, "fig_samples.png")
    fig.save(out, dpi=(300, 300))

    note = os.path.join(ROOT, "fig_samples_NOTE.txt")
    with open(note, "w") as fh:
        fh.write(
            "fig_samples.png — selection provenance (no cherry-picking)\n"
            f"seed: random.Random({SEED}), single instance\n"
            "draw order: real[missing_hole, short, spur], then "
            "generated[missing_hole, short, spur]; rng.choice(sorted(pool))\n"
            "real pool: PCB-cropped-tight/all, un-augmented originals only "
            "(no rotation_/l_ variants)\n"
            "generated pool: synth_b100 (b100 conditional generator)\n"
            "tiles 256x256 as-is; 300 dpi\n\nchosen files:\n")
        for (row, cls), p in picks.items():
            fh.write(f"  {row:9s} {cls:13s} {os.path.relpath(p, ROOT)}\n")
    print(out)
    print(note)
    for (row, cls), p in picks.items():
        print(f"  {row:9s} {cls:13s} {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
