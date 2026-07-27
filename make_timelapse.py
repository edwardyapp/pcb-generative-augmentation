"""Record one full VQ-VAE-2 sampling run as a timelapse video for slides.

Runs a single-sample generation and captures a snapshot after every completed
row of codes:

  Phase 1 -- top prior (32x32): the code-index map rendered as a viridis grid,
             current row highlighted.
  Phase 2 -- bottom prior (64x64): the image decoded from the full top codes +
             the partial bottom codes so far, with the not-yet-sampled region
             blacked out (each bottom code = one 4x4 patch of the 256x256
             image), shown large next to the small code grid.

Frames are 1920x1080 on a dark background with projector-sized text. Encodes an
H.264 mp4 (yuv420p, even dims -> PowerPoint-safe) plus a GIF fallback.

Reuses the model-loading / sampling / decoding helpers from generate_pool.py
(without changing its behaviour). One seed per run:

    python make_timelapse.py --seed 0 --out demo_timelapse/
"""

import argparse
import os
import shutil
import subprocess

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

from generate_pool import (
    load_models,
    tensor_to_pil,
    TOP_SHAPE,
    BOTTOM_SHAPE,
)

# 256x256 decoded image / 64x64 bottom grid -> each bottom code is a 4x4 patch.
PATCH = 256 // BOTTOM_SHAPE[0]

DARK_BG = "#0d0d0d"
FG = "#f0f0f0"
MUTED = "#9aa0a6"
HIGHLIGHT = "#ffcc00"

# Pacing (seconds) and output frame rate.
PHASE1_SEC = 6
PHASE2_SEC = 14
HOLD_SEC = 3
FPS = 30


# --------------------------------------------------------------------------- #
# Frame rendering
# --------------------------------------------------------------------------- #
def _new_fig():
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)  # -> exactly 1920x1080
    fig.patch.set_facecolor(DARK_BG)
    return fig


def _save(fig, path):
    fig.savefig(path, dpi=100, facecolor=DARK_BG)
    plt.close(fig)


def _style_grid_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#333333")


def _draw_progress(ax, cur_row, n):
    """Dim the un-sampled rows and outline the current (just-completed) row."""
    if cur_row < n - 1:
        ax.add_patch(
            Rectangle(
                (-0.5, cur_row + 0.5), n, (n - 1) - cur_row,
                facecolor=DARK_BG, alpha=0.72, edgecolor="none",
            )
        )
    ax.add_patch(
        Rectangle((-0.5, cur_row - 0.5), n, 1, fill=False, edgecolor=HIGHLIGHT, lw=3)
    )


def render_top_frame(grid, cur_row, path):
    n = grid.shape[0]
    fig = _new_fig()
    fig.text(0.5, 0.93, "Stage 1: top prior — 32×32 latent codes",
             ha="center", va="center", color=FG, fontsize=42, fontweight="bold")
    fig.text(0.5, 0.855, f"row {cur_row + 1} / {n}",
             ha="center", va="center", color=MUTED, fontsize=26)
    ax = fig.add_axes([0.34, 0.06, 0.32, 0.74])
    ax.imshow(grid, cmap="viridis", vmin=0, vmax=511,
              interpolation="nearest", aspect="equal")
    _draw_progress(ax, cur_row, n)
    _style_grid_ax(ax)
    _save(fig, path)


def render_bottom_frame(masked_img, bgrid, rows_completed, path):
    n = bgrid.shape[0]
    fig = _new_fig()
    fig.text(0.5, 0.93, "Stage 2: bottom prior — 64×64 codes → decoder",
             ha="center", va="center", color=FG, fontsize=40, fontweight="bold")
    fig.text(0.5, 0.86, f"row {rows_completed} / {n}",
             ha="center", va="center", color=MUTED, fontsize=24)

    ax_img = fig.add_axes([0.05, 0.06, 0.54, 0.74])
    ax_img.imshow(masked_img, aspect="equal", interpolation="nearest")
    ax_img.set_title("decoded image (256×256)", color=FG, fontsize=24, pad=12)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for s in ax_img.spines.values():
        s.set_color("#333333")

    ax_g = fig.add_axes([0.65, 0.20, 0.30, 0.46])
    ax_g.imshow(bgrid, cmap="viridis", vmin=0, vmax=511,
                interpolation="nearest", aspect="equal")
    ax_g.set_title("bottom codes (64×64)", color=FG, fontsize=22, pad=12)
    _draw_progress(ax_g, rows_completed - 1, n)
    _style_grid_ax(ax_g)
    _save(fig, path)


def render_final_frame(img_arr, path):
    fig = _new_fig()
    fig.text(0.5, 0.93, "Generated PCB defect sample",
             ha="center", va="center", color=FG, fontsize=46, fontweight="bold")
    ax = fig.add_axes([0.30, 0.05, 0.40, 0.80])
    ax.imshow(img_arr, aspect="equal", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#333333")
    _save(fig, path)


# --------------------------------------------------------------------------- #
# Sampling with per-row snapshots (mirrors generate_pool.sample_model)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_timelapse(models, seed, temperature=1.0):
    device = models["device"]
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    top, bottom, vqvae = models["top"], models["bottom"], models["vqvae"]

    # ---- Phase 1: top prior ----
    trow = torch.zeros(1, *TOP_SHAPE, dtype=torch.int64, device=device)
    cache = {}
    top_snaps = []
    for i in tqdm(range(TOP_SHAPE[0]), desc=f"seed {seed} top ", leave=False):
        for j in range(TOP_SHAPE[1]):
            out, cache = top(trow[:, : i + 1, :], condition=None, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            trow[0, i, j] = torch.multinomial(prob, 1).squeeze(-1)
        top_snaps.append(trow[0].cpu().numpy().copy())
    top_sample = trow

    # ---- Phase 2: bottom prior (decode partial state after each row) ----
    brow = torch.zeros(1, *BOTTOM_SHAPE, dtype=torch.int64, device=device)
    cache = {}
    bottom_snaps = []
    for i in tqdm(range(BOTTOM_SHAPE[0]), desc=f"seed {seed} bot ", leave=False):
        for j in range(BOTTOM_SHAPE[1]):
            out, cache = bottom(brow[:, : i + 1, :], condition=top_sample, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            brow[0, i, j] = torch.multinomial(prob, 1).squeeze(-1)

        rows_completed = i + 1
        decoded = vqvae.decode_code(top_sample, brow).clamp(-1, 1)
        arr = np.array(tensor_to_pil(decoded[0]))

        # Mask built from the code grid: sampled codes -> 4x4 visible patches.
        sampled = np.zeros(BOTTOM_SHAPE, dtype=bool)
        sampled[:rows_completed, :] = True
        mask = np.repeat(np.repeat(sampled, PATCH, axis=0), PATCH, axis=1)
        masked = arr.copy()
        masked[~mask] = 0

        bottom_snaps.append((masked, brow[0].cpu().numpy().copy(), rows_completed))

    # Final = fully-sampled decode (identical to the last bottom snapshot, unmasked).
    final_decoded = vqvae.decode_code(top_sample, brow).clamp(-1, 1)
    final_arr = np.array(tensor_to_pil(final_decoded[0]))
    return top_snaps, bottom_snaps, final_arr


# --------------------------------------------------------------------------- #
# Frame sequencing + ffmpeg encode
# --------------------------------------------------------------------------- #
def _run_ffmpeg(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed:\n" + proc.stdout.decode("utf-8", "replace")[-2000:]
        )


def build_and_encode(top_snaps, bottom_snaps, final_arr, out_dir, seed, fps=FPS):
    frames_dir = os.path.join(out_dir, f"_frames_seed{seed}")
    seq_dir = os.path.join(frames_dir, "seq")
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(seq_dir, exist_ok=True)

    # Render each unique logical frame once.
    top_paths = []
    for k, grid in enumerate(top_snaps):
        p = os.path.join(frames_dir, f"t_{k:03d}.png")
        render_top_frame(grid, k, p)
        top_paths.append(p)

    bot_paths = []
    for k, (masked, bgrid, rc) in enumerate(bottom_snaps):
        p = os.path.join(frames_dir, f"b_{k:03d}.png")
        render_bottom_frame(masked, bgrid, rc, p)
        bot_paths.append(p)

    final_path = os.path.join(frames_dir, "final.png")
    render_final_frame(final_arr, final_path)

    # Expand into a constant-rate sequence via hardlinks (falls back to copy).
    counter = {"n": 0}

    def emit(src):
        dst = os.path.join(seq_dir, f"{counter['n']:06d}.png")
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy(src, dst)
        counter["n"] += 1

    p1, p2, hold = int(PHASE1_SEC * fps), int(PHASE2_SEC * fps), int(HOLD_SEC * fps)
    for t in range(p1):
        emit(top_paths[min(len(top_paths) - 1, t * len(top_paths) // p1)])
    for t in range(p2):
        emit(bot_paths[min(len(bot_paths) - 1, t * len(bot_paths) // p2)])
    for _ in range(hold):
        emit(final_path)

    seq_glob = os.path.join(seq_dir, "%06d.png")
    mp4 = os.path.join(out_dir, f"timelapse_seed{seed}.mp4")
    gif = os.path.join(out_dir, f"timelapse_seed{seed}.gif")

    # H.264 mp4: yuv420p + forced even dims for PowerPoint compatibility.
    _run_ffmpeg([
        "ffmpeg", "-y", "-framerate", str(fps), "-i", seq_glob,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-movflags", "+faststart", mp4,
    ])

    # GIF fallback: downscaled, palette-optimised, infinite loop.
    _run_ffmpeg([
        "ffmpeg", "-y", "-framerate", str(fps), "-i", seq_glob,
        "-vf",
        "fps=15,scale=960:-2:flags=lanczos,"
        "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        "-loop", "0", gif,
    ])

    shutil.rmtree(frames_dir)
    return mp4, gif


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="demo_timelapse/")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    os.makedirs(args.out, exist_ok=True)

    print(f"[seed {args.seed}] loading models on {args.device} ...")
    models = load_models(device=args.device)

    print(f"[seed {args.seed}] sampling (single image, capturing every row) ...")
    top_snaps, bottom_snaps, final_arr = run_timelapse(models, args.seed, args.temp)

    print(f"[seed {args.seed}] rendering "
          f"{len(top_snaps) + len(bottom_snaps) + 1} frames + encoding ...")
    mp4, gif = build_and_encode(top_snaps, bottom_snaps, final_arr, args.out,
                                args.seed, args.fps)

    print(f"[seed {args.seed}] done:")
    print(f"    mp4: {os.path.abspath(mp4)}")
    print(f"    gif: {os.path.abspath(gif)}")


if __name__ == "__main__":
    main()
