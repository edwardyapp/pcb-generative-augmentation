"""fig_samples.png / .pdf — real vs b100-generated defect crops, 2x3, for the paper.

Top row: real defect-centred tight crops (PCB-cropped-tight/all, 256x256,
un-augmented originals only: no rotation_/l_ variants). Bottom row: b100
conditional generator samples (synth_b100), same classes.

NO CHERRY-PICKING: one random.Random(0) instance; picks drawn with
rng.choice(sorted(pool)) in the fixed order real:[missing_hole, short, spur]
then generated:[missing_hole, short, spur]. Chosen files recorded in
fig_samples_NOTE.txt.

B/W-print safe: black text on white, no colour-coded annotations.

Labels are sized against the PRINTED width (PRINT_WIDTH_IN), to sit at 10 pt
body text. A 256 px tile is the hard limit in both directions and fit_font caps
each label to it, returning the largest size that fits:
  - COLUMNS: "missing_hole" on one line is 388 px at 58 px against a 256 px
    tile, so column labels wrap at the underscore ("missing" / "hole").
    "missing" is then 222 px, so 58 px stands: 9.96 pt, i.e. 10 pt as intended.
  - ROWS: horizontally, "generated" is 300 px at 58 px against an 84 px gutter,
    so row labels are rotated 90 deg. Rotation trades the width limit for a
    height one -- the label runs along the 256 px tile height, where
    "generated" is still 300 px at 58 px. Capped at 49 px = 8.4 pt, the largest
    that fits. Rows cannot reach 10 pt at this print width; the note file
    records the actual value rather than the requested one.

assert_no_clipping() fails the build if any ink reaches the canvas edge, so a
cap that is set too high can never silently truncate a label.

Tiles are pasted 1:1 at native 256x256 and are never resampled; changing label
sizes cannot alter tile pixels.

Output: fig_samples.png (600 dpi), fig_samples.pdf (vector text, embedded
rasters) and fig_samples_NOTE.txt in project root.
"""
import os
import random
import re

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in analysis/)
CLASSES = ["missing_hole", "short", "spur"]
TILE, GAP = 256, 4
SEED = 0
DPI = 600

# Label sizing is driven by the PRINTED size, not by a pixel constant: a label
# of N px in a W px-wide figure placed at PRINT_WIDTH_IN inches occupies
# N/W * PRINT_WIDTH_IN * 72 points. TARGET_PT is the body-text size to match.
#
# PRINT_WIDTH_IN is MEASURED, not assumed: 2.500 in = 2286000 EMU, read from
# wp:extent in the submitted .docx (which is not in this repository). It is the
# single input that sets both label sizes -- change it and solve_px re-solves.
#
# The pixel size is solved rather than hard-coded because W itself depends on
# the row font (the gutter is sized to the rotated label's height), so px and W
# are mutually dependent. solve_px iterates the fixed point; it converges in
# two passes because W moves only a few px across the plausible range.
#
# fit_font then caps each label so nothing overruns a 256 px tile, returning the
# largest size that fits. At 2.5 in the solved size is ~46 px and NEITHER cap
# binds -- at 2.0 in the row cap did bind at 49 px, holding rows to 8.4 pt.
PRINT_WIDTH_IN = 2.5
TARGET_PT = 10.0
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


def pt(px_size, fig_w):
    """Point size a px-tall label occupies when the figure is printed at
    PRINT_WIDTH_IN inches wide."""
    return px_size / fig_w * PRINT_WIDTH_IN * 72.0


def fit_font(strings, limit, target):
    """Largest size <= target at which every string measures <= limit px along
    its layout axis. Guards against silent clipping."""
    d = _probe()
    for s in range(target, 5, -1):
        f = load_font(s)
        if max(text_size(d, t, f)[0] for t in strings) <= limit:
            return s
    raise SystemExit(f"no font <= {target}px fits {strings} in {limit}px")


def figure_width(row_px):
    """Figure width in px. Depends on the ROW font only: the gutter is sized to
    the rotated label's height. Column font affects height, not width."""
    d = _probe()
    f = load_font(row_px)
    row_h = max(text_size(d, r, f)[1] for r in ("real", "generated"))
    return MARGIN + (row_h + GUTTER_PAD) + 3 * TILE + 2 * GAP + MARGIN


def solve_px(target_pt):
    """Pixel size whose printed size is target_pt at PRINT_WIDTH_IN.

    px = target_pt * W / (PRINT_WIDTH_IN * 72), but W depends on px through the
    gutter, so iterate to the fixed point."""
    px = 50
    for _ in range(8):
        capped = fit_font(["real", "generated"], TILE, max(6, int(round(px))))
        nxt = target_pt * figure_width(capped) / (PRINT_WIDTH_IN * 72.0)
        if int(round(nxt)) == int(round(px)):
            break
        px = nxt
    return max(6, int(round(px)))


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
    # Both labels want the same printed size, so both start from one solved
    # pixel size; fit_font then caps each against the 256 px tile.
    want_px = solve_px(TARGET_PT)
    col_px = fit_font([w for c in CLASSES for w in wrap(c)], TILE, want_px)
    row_px = fit_font(["real", "generated"], TILE, want_px)
    f_col, f_row = load_font(col_px), load_font(row_px)

    line_h = max(text_size(d, "Ag", f_col)[1] for _ in (0,))
    lead = int(round(col_px * 0.18))
    n_lines = max(len(wrap(c)) for c in CLASSES)
    header_h = n_lines * line_h + (n_lines - 1) * lead + HEADER_PAD

    row_w = max(text_size(d, r, f_row)[0] for r in ("real", "generated"))
    # extent the row label WOULD have at the solved size, for the note
    f_want = load_font(want_px)
    row_w_target = max(text_size(d, r, f_want)[0] for r in ("real", "generated"))
    row_h = max(text_size(d, r, f_row)[1] for r in ("real", "generated"))
    gutter_w = row_h + GUTTER_PAD          # rotated: height becomes width

    W = MARGIN + gutter_w + 3 * TILE + 2 * GAP + MARGIN
    H = MARGIN + header_h + 2 * TILE + GAP + MARGIN
    return dict(W=W, H=H, header_h=header_h, gutter_w=gutter_w,
                line_h=line_h, lead=lead, row_w=row_w, row_h=row_h,
                x0=MARGIN + gutter_w, y0=MARGIN + header_h,
                f_col=f_col, f_row=f_row, col_px=col_px, row_px=row_px,
                row_w_target=row_w_target, want_px=want_px)


def assert_no_clipping(fig):
    """Fail loudly if any ink reaches the canvas edge. An earlier revision drew
    a rotated row label 331 px long into a 256 px row and silently clipped the
    "g" off "generated"; the metrics all looked fine. This catches that."""
    W, H = fig.size
    px = fig.load()
    ink = lambda x, y: sum(px[x, y][:3]) < 600
    edges = {"top": [(x, 0) for x in range(W)],
             "bottom": [(x, H - 1) for x in range(W)],
             "left": [(0, y) for y in range(H)],
             "right": [(W - 1, y) for y in range(H)]}
    hits = {k: sum(1 for p in v if ink(*p)) for k, v in edges.items()}
    bad = {k: n for k, n in hits.items() if n}
    if bad:
        raise SystemExit(f"content clipped at canvas edge: {bad}")


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

    assert_no_clipping(fig)
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

    col_note = (f"{g['col_px']} px = {pt(g['col_px'], g['W']):.1f} pt "
                + ("(tile-width cap not binding)" if g["col_px"] == g["want_px"]
                   else f"CAPPED from {g['want_px']} px to fit the {TILE} px tile"))
    row_note = (f"{g['row_px']} px = {pt(g['row_px'], g['W']):.1f} pt "
                + (f"(rotated label spans {g['row_w']} px against a {TILE} px "
                   "row; cap not binding)" if g["row_px"] == g["want_px"]
                   else f"CAPPED: at {g['want_px']} px the rotated label would "
                        f"span {g['row_w_target']} px against a {TILE} px row"))

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
            f"figure {g['W']}x{g['H']} px\n"
            f"placed at {PRINT_WIDTH_IN:.3f} in wide — MEASURED from wp:extent "
            "in the submitted .docx (2286000 EMU), not assumed\n"
            f"labels target {TARGET_PT:.0f} pt body text at that width; solved "
            f"size {g['want_px']} px\n"
            f"column labels: wrap at the underscore, {col_note}\n"
            f"row labels: rotated 90 deg, {row_note}\n"
            f"if the figure is placed at another width, multiply both by "
            f"(width / {PRINT_WIDTH_IN:.3f})\n"
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
