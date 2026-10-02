# PCB Defect Generation — Stage Demo

Everything for the live VQ-VAE-2 demo. Three new files, no training code or
checkpoints touched:

| file | what it does |
|------|--------------|
| `legacy/generate_pool.py` | pre-generates a pool of sample images (run tonight) |
| `legacy/demo_app.py`      | Gradio app for the stage (run tomorrow) |
| `legacy/make_timelapse.py`| renders the sampling-process clips for your slides |

**Checkpoints used** (auto-detected, latest epoch): `checkpoint/vqvae_560.pt`,
`checkpoint/pixelsnail_top_357.pt`, `checkpoint/pixelsnail_bottom_best.pt`.
Runs on CUDA (`cuda:0` = the RTX 4090) using the `python` on your PATH.

---

## 1. Tonight — generate the image pool

```bash
python legacy/generate_pool.py --n 300 --batch 16 --temp 1.0 --out demo_pool/
```

**Time estimate:** measured **~26 s/image** on your 4090 (batch 8 → 26.1 s/img,
batch 16 → 25.9 s/img; the autoregressive bottom prior dominates, so throughput
is basically flat across batch size).

- Full 300 from scratch ≈ **2 h 10 m**.
- The pool already holds **24 images** from testing, and the script is
  **resumable** (it skips files that already exist), so tonight's run will
  generate the remaining ~276 in **≈ 2 h**.

Kick it off before you go to sleep. Notes:
- If it hits CUDA OOM it auto-halves the batch and retries — you don't have to
  babysit it.
- Progress + ETA print after every batch. Safe to Ctrl-C and re-run; it resumes.
- Running on `--device cuda:1` (the RTX 5090, 32 GB) is ~2× faster: measured
  **~12.8 s/image**, so the remaining ~276 finish in **≈ 55 min**. That's how
  the run was actually launched (`--device cuda:1`).

---

## 2. Tomorrow — launch the app

```bash
python legacy/demo_app.py
```

Then open **http://127.0.0.1:7861** in a browser (it does not auto-open, and
there is no public share link — fully offline, analytics disabled).

- Big **"Generate defects"** button → shows 4 random pool images in a 2×2
  gallery with a caption like `samples 042, 137, 201, 288 of 300`. Instant.
- Loads dark mode automatically and fills the gallery on open.
- Collapsed **"true live sampling (slow)"** section runs one real batch through
  the models — leave it closed on stage unless you want to show it live (the
  first click also loads the models, so it takes a minute+).

Put the browser in fullscreen (F11) before you present.

---

## 3. Backup screen capture (do this tonight after the pool finishes)

Live demos fail; have a recording ready to drop into your slides / play if the
laptop misbehaves on stage.

1. Launch the app (`python legacy/demo_app.py`), fullscreen the browser.
2. Record ~30–60 s of you clicking **Generate defects** a bunch of times:
   - **Ubuntu 24.04 (GNOME):** built-in recorder — `Ctrl + Alt + Shift + R`
     starts/stops; the clip lands in `~/Videos/Screencasts/`.
   - or **ffmpeg** (X11): `ffmpeg -video_size 1920x1080 -f x11grab -i :0.0 -c:v libx264 -pix_fmt yuv420p demo_backup.mp4`
3. Keep the clip on the same USB/folder as your slides.

The `legacy/make_timelapse.py` clips below are also good standalone backups — worst
case, you can just play those.

---

## Timelapse clips for your slides

Three seeds rendered so you can pick the nicest (1920×1080, H.264 mp4 +
GIF fallback, ~23 s each: 6 s top prior → 14 s bottom prior/decode → 3 s hold):

```
demo_timelapse/timelapse_seed0.mp4   (+ .gif)
demo_timelapse/timelapse_seed1.mp4   (+ .gif)
demo_timelapse/timelapse_seed2.mp4   (+ .gif)
```

The mp4s are yuv420p with even dimensions, so they drop straight into
PowerPoint. To render more seeds: `python legacy/make_timelapse.py --seed 3`.
