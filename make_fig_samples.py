"""fig_samples.png / .pdf — real vs b100-generated defect crops, 2x3, for the paper.

Top row: real defect-centred tight crops (PCB-cropped-tight/all, 256x256,
un-augmented originals only: no rotation_/l_ variants). Bottom row: b100
conditional generator samples (synth_b100), same classes.

NO CHERRY-PICKING: one random.Random(0) instance; picks drawn with
rng.choice(sorted(pool)) in the fixed order real:[missing_hole, short, spur]
then generated:[missing_hole, short, spur]. Chosen files recorded in
fig_samples_NOTE.txt.

B/W-print safe: black text on white, no colour-coded annotations.

Labels are sized for legibility at ~2-inch print width, targeting ~4x the
original 17/16 px. Neither axis reaches a full 4x, because a 256 px tile is the
hard limit in both directions and fit_font caps each to it:
  - COLUMNS: "missing_hole" on one line is 441 px at 4x against a 256 px tile,
    so column labels wrap at the underscore ("missing" / "hole"). "missing" is
    253 px at 66 and 257 px at 67 -> 66 px, i.e. 3.9x.
  - ROWS: horizontally, "generated" is 331 px against an 84 px gutter (largest
    single-line fit 16 px = 1.0x), so row labels are rotated 90 deg. Rotation
    trades the width limit for a height one: the label now runs along the
    256 px tile height, where "generated" is 331 px at 4x and would clip. The
    cap is 49 px, i.e. 3.1x.

Net effect: the gutter shrinks 92 -> 59 px and the tiles' share of the figure
width RISES, 87.3% -> 91.5%. At 2-inch print width the labels are ~11.3 pt
(columns) and ~8.4 pt (rows), against ~2.8 / ~2.6 pt before.

Tiles are pasted 1:1 at native 256x256 and are never resampled; enlarging the
labels cannot alter tile pixels.

Output: fig_samples.png (600 dpi), fig_samples.pdf (vector text, embedded
rasters) and fig_samples_NOTE.txt in project root.
"""
import os
import random
import re

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["missing_hole", "short", "spur"]
TILE, GAP = 256, 4
SEED = 0
DPI = 600

# Target ~4x the original 17/16 px, then cap each so nothing overruns a 256 px
# tile. Both caps are computed (see fit_font) rather than hard-coded, so the
# figure cannot silently clip if a class name or font ever changes:
#   - column labels wrap at the underscore; the widest token must fit the tile
#     WIDTH. "missing" is 253 px at 66 and 257 px at 67 -> cap 66 (3.9x).
#   - row labels are rotated 90 deg, so their length runs along the tile
#     HEIGHT. "generated" is 331 px at 64, which overran the 256 px row and
#     clipped the "g" -> cap 49 (3.1x).
TARGET_COL, TARGET_ROW = 68, 64
MARGIN = 2          # was 6; trimmed so the enlarged labels cost the tiles less
GUTTER_PAD = 12     # around the rotated row label
HEADER_PAD = 8      # between the column label block and the tile top

FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf")

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
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def font_path():
    for p in FONT_PATHS:
        if os.path.exists(p):
            return p
    return None


def wrap(cls):
    """Column label lines: split at the underscore so 4x text fits the tile."""
    return cls.split("_")


def _probe():
    return ImageDraw.Draw(Image.new("RGB", (8, 8)))


def fit_font(strings, limit, target):
    """Largest size <= target at which every string measures <= limit px along
    its layout axis. Guards against silent clipping."""
    d = _probe()
    for s in range(target, 5, -1):
        f = load_font(s)
        if max(text_size(d, t, f)[0] for t in strings) <= limit:
            return s
    raise SystemExit(f"no font <= {target}px fits {strings} in {limit}px")


def text_size(draw, s, font):
    b = draw.textbbox((0, 0), s, font=font)
    return b[2] - b[0], b[3] - b[1]


def render_text(s, font, fill="black"):
    """Tightly-cropped RGBA image of `s`, so placement is not offset by the
    font's internal bearing."""
    d0 = _probe()
    b = d0.textbbox((0, 0), s, font=font)
    im = Image.new("RGBA", (b[2] - b[0], b[3] - b[1]), (255, 255, 255, 0))
    ImageDraw.Draw(im).text((-b[0], -b[1]), s, font=font, fill=fill)
    return im


def geometry():
    """Shared by the PNG and PDF paths so the two renderings agree."""
    d = _probe()
    # column labels lie along the tile width; rotated row labels along its height
    col_px = fit_font([w for c in CLASSES for w in wrap(c)], TILE, TARGET_COL)
    row_px = fit_font(["real", "generated"], TILE, TARGET_ROW)
    f_col, f_row = load_font(col_px), load_font(row_px)

    line_h = max(text_size(d, "Ag", f_col)[1] for _ in (0,))
    lead = int(round(col_px * 0.18))
    n_lines = max(len(wrap(c)) for c in CLASSES)
    header_h = n_lines * line_h + (n_lines - 1) * lead + HEADER_PAD

    row_w = max(text_size(d, r, f_row)[0] for r in ("real", "generated"))
    row_h = max(text_size(d, r, f_row)[1] for r in ("real", "generated"))
    gutter_w = row_h + GUTTER_PAD          # rotated: height becomes width

    W = MARGIN + gutter_w + 3 * TILE + 2 * GAP + MARGIN
    H = MARGIN + header_h + 2 * TILE + GAP + MARGIN
    return dict(W=W, H=H, header_h=header_h, gutter_w=gutter_w,
                line_h=line_h, lead=lead, row_w=row_w, row_h=row_h,
                x0=MARGIN + gutter_w, y0=MARGIN + header_h,
                f_col=f_col, f_row=f_row, col_px=col_px, row_px=row_px)


def draw_png(picks, g):
    fig = Image.new("RGB", (g["W"], g["H"]), "white")
    draw = ImageDraw.Draw(fig)

    # column labels: wrapped, centred on the tile, bottom-aligned to the tiles
    for j, cls in enumerate(CLASSES):
        cx = g["x0"] + j * (TILE + GAP)
        lines = wrap(cls)
        block_h = len(lines) * g["line_h"] + (len(lines) - 1) * g["lead"]
        top = MARGIN + g["header_h"] - HEADER_PAD - block_h
        for k, ln in enumerate(lines):
            w, _ = text_size(draw, ln, g["f_col"])
            draw.text((cx + (TILE - w) / 2, top + k * (g["line_h"] + g["lead"])),
                      ln, fill="black", font=g["f_col"])

    # row labels: rotated 90 deg, centred in the gutter and on the row
    for i, row in enumerate(("real", "generated")):
        ry = g["y0"] + i * (TILE + GAP)
        lab = render_text(row, g["f_row"]).rotate(90, expand=True)
        fig.paste(lab,
                  (MARGIN + (g["gutter_w"] - lab.width) // 2,
                   ry + (TILE - lab.height) // 2),
                  lab)
        for j, cls in enumerate(CLASSES):
            im = Image.open(picks[(row, cls)]).convert("RGB")
            if im.size != (TILE, TILE):
                raise SystemExit(f"{picks[(row, cls)]} is {im.size}, not 256x256")
            fig.paste(im, (g["x0"] + j * (TILE + GAP), ry))

    out = os.path.join(ROOT, "fig_samples.png")
    fig.save(out, dpi=(DPI, DPI))
    return out


def draw_pdf(picks, g):
    """Same layout via matplotlib so labels stay vector text in the PDF while
    the tiles are embedded as rasters."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42        # TrueType, not outlines
    matplotlib.rcParams["pdf.compression"] = 6
    fp = font_path()
    fprop = matplotlib.font_manager.FontProperties(fname=fp) if fp else None

    W, H = g["W"], g["H"]
    px = 72.0 / DPI                                  # px -> pt at this dpi
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor("white")

    def fx(x):
        return x / W

    def fy(y):
        return 1.0 - y / H                            # PIL y-down -> fig y-up

    for i, row in enumerate(("real", "generated")):
        ry = g["y0"] + i * (TILE + GAP)
        for j, cls in enumerate(CLASSES):
            ax = fig.add_axes([fx(g["x0"] + j * (TILE + GAP)),
                               fy(ry + TILE), TILE / W, TILE / H])
            ax.imshow(Image.open(picks[(row, cls)]).convert("RGB"),
                      interpolation="none", resample=False)
            ax.set_axis_off()
        fig.text(fx(MARGIN + g["gutter_w"] / 2), fy(ry + TILE / 2), row,
                 rotation=90, ha="center", va="center", color="black",
                 fontsize=g["row_px"] * px, fontproperties=fprop)

    for j, cls in enumerate(CLASSES):
        cx = g["x0"] + j * (TILE + GAP) + TILE / 2
        lines = wrap(cls)
        block_h = len(lines) * g["line_h"] + (len(lines) - 1) * g["lead"]
        top = MARGIN + g["header_h"] - HEADER_PAD - block_h
        for k, ln in enumerate(lines):
            yc = top + k * (g["line_h"] + g["lead"]) + g["line_h"] / 2
            fig.text(fx(cx), fy(yc), ln, ha="center", va="center",
                     color="black", fontsize=g["col_px"] * px, fontproperties=fprop)

    out = os.path.join(ROOT, "fig_samples.pdf")
    fig.savefig(out, format="pdf", dpi=DPI, facecolor="white")
    plt.close(fig)
    return out


def main():
    rng = random.Random(SEED)
    picks = {}   # (row, cls) -> path, drawn in documented fixed order
    for row, pool_fn in (("real", real_pool), ("generated", gen_pool)):
        for cls in CLASSES:
            pool = pool_fn(cls)
            if not pool:
                raise SystemExit(f"empty pool: {row}/{cls}")
            picks[(row, cls)] = rng.choice(pool)

    g = geometry()
    out = draw_png(picks, g)
    pdf = draw_pdf(picks, g)

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
            f"tiles 256x256 as-is; {DPI} dpi\n"
            "also written: fig_samples.pdf (same layout, same tiles; vector "
            "text, embedded rasters)\n"
            "labels enlarged for 2-inch print width: column labels wrap at the "
            f"underscore ({g['col_px']} px), row labels rotated 90 deg "
            f"({g['row_px']} px); each capped to fit the 256 px tile\n"
            "\nchosen files:\n")
        for (row, cls), p in picks.items():
            fh.write(f"  {row:9s} {cls:13s} {os.path.relpath(p, ROOT)}\n")
    print(out)
    print(pdf)
    print(note)
    for (row, cls), p in picks.items():
        print(f"  {row:9s} {cls:13s} {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
