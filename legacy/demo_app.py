"""Gradio app for the live PCB-defect-generation demo.

Two modes:
  * Instant: one big "Generate defects" button that shows 4 random images from
    the pre-generated pool (demo_pool/). Use this on stage -- it's instant.
  * "true live sampling (slow)": a collapsed section that runs one real batch
    through the VQ-VAE-2 + PixelSNAIL priors. Off by default; takes a while.

Fully offline: binds to 127.0.0.1, no share link, analytics disabled.

    python legacy/demo_app.py                    # defaults: pool demo_pool/, port 7861
    python legacy/demo_app.py --pool demo_pool/ --device cuda
"""

import argparse
import os
import random
import re
import subprocess
import sys

# Must be set before gradio is imported.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "0"

try:
    import gradio as gr
except ImportError:
    print("gradio not found -- installing (pip install gradio) ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
    import gradio as gr


DEFAULT_POOL = "demo_pool/"
DEFAULT_DEVICE = "cuda"
DEFAULT_PORT = 7861
LIVE_BATCH = 4
LIVE_TEMP = 1.0

# Filled from argparse in main(); referenced by the event handlers.
CONFIG = {"pool": DEFAULT_POOL, "device": DEFAULT_DEVICE, "temp": LIVE_TEMP}

# Lazily-loaded models for the optional live-sampling path.
_MODELS = {}

_NUM_RE = re.compile(r"(\d+)")


def _index_of(path):
    m = _NUM_RE.findall(os.path.basename(path))
    return int(m[-1]) if m else -1


def list_pool():
    """Return sorted list of PNG paths in the pool directory."""
    pool_dir = CONFIG["pool"]
    if not os.path.isdir(pool_dir):
        return []
    files = [
        os.path.join(pool_dir, f)
        for f in os.listdir(pool_dir)
        if f.lower().endswith(".png")
    ]
    return sorted(files, key=_index_of)


def show_random():
    """Pick 4 random pool images for the 2x2 gallery + a caption."""
    pool = list_pool()
    if not pool:
        return [], (
            f"### ⚠ no images in `{CONFIG['pool']}` — run "
            "`python legacy/generate_pool.py` first"
        )

    k = min(4, len(pool))
    picks = sorted(random.sample(pool, k), key=_index_of)
    nums = ", ".join(f"{_index_of(p):03d}" for p in picks)
    caption = f"### samples {nums} of {len(pool)}"
    return picks, caption


def live_sample():
    """Run one real batch through the models. Slow; loads models on first use."""
    try:
        from generate_pool import (
            load_models,
            sample_decoded,
            tensor_to_pil,
            upscale,
        )
    except Exception as e:  # noqa: BLE001
        return [], f"### live sampling unavailable: {e}"

    if "m" not in _MODELS:
        _MODELS["m"] = load_models(device=CONFIG["device"])

    decoded = sample_decoded(_MODELS["m"], LIVE_BATCH, CONFIG["temp"])
    imgs = [upscale(tensor_to_pil(decoded[i])) for i in range(decoded.shape[0])]
    return imgs, (
        f"### {len(imgs)} freshly sampled images "
        f"(temp {CONFIG['temp']}, device {CONFIG['device']})"
    )


# Force dark mode on load (projector-friendly), regardless of browser default.
FORCE_DARK_JS = """
() => {
  const url = new URL(window.location.href);
  if (url.searchParams.get('__theme') !== 'dark') {
    url.searchParams.set('__theme', 'dark');
    window.location.href = url.href;
  }
}
"""

CSS = """
.gradio-container { max-width: 100% !important; }
#gen-btn {
  font-size: 2.2rem !important;
  font-weight: 800 !important;
  padding: 32px !important;
  margin: 8px 0 4px 0 !important;
}
#main-gallery { min-height: 72vh; }
.big-caption { font-size: 1.5rem !important; text-align: center; margin: 6px 0; }
#title h1 { text-align: center; }
footer { display: none !important; }
"""


def build_app():
    with gr.Blocks(title="PCB defect generation — VQ-VAE-2") as demo:
        gr.Markdown("# PCB defect generation · VQ-VAE-2", elem_id="title")

        gen_btn = gr.Button("Generate defects", variant="primary", elem_id="gen-btn")
        caption = gr.Markdown("### click **Generate defects**", elem_classes=["big-caption"])
        gallery = gr.Gallery(
            label=None,
            show_label=False,
            columns=2,
            rows=2,
            height="72vh",
            object_fit="contain",
            allow_preview=True,
            elem_id="main-gallery",
        )

        gen_btn.click(show_random, inputs=None, outputs=[gallery, caption])
        # Populate immediately when the page opens.
        demo.load(show_random, inputs=None, outputs=[gallery, caption])

        with gr.Accordion("true live sampling (slow)", open=False):
            gr.Markdown(
                "Runs one real batch through the VQ-VAE-2 + PixelSNAIL priors. "
                "The first run also loads the models, so it can take a while."
            )
            live_btn = gr.Button(f"Run one live batch ({LIVE_BATCH} images)")
            live_caption = gr.Markdown("")
            live_gallery = gr.Gallery(
                show_label=False,
                columns=2,
                rows=2,
                height="60vh",
                object_fit="contain",
                allow_preview=True,
            )
            live_btn.click(live_sample, inputs=None, outputs=[live_gallery, live_caption])

    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=str, default=DEFAULT_POOL)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--temp", type=float, default=LIVE_TEMP)
    args = parser.parse_args()

    CONFIG["pool"] = args.pool
    CONFIG["device"] = args.device
    CONFIG["temp"] = args.temp

    print(f"Serving pool from: {os.path.abspath(CONFIG['pool'])}")
    print(f"Images currently in pool: {len(list_pool())}")

    demo = build_app()
    # In Gradio 6, theme/css/js are passed to launch() rather than Blocks().
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        inbrowser=False,
        theme=gr.themes.Base(),
        css=CSS,
        js=FORCE_DARK_JS,
    )


if __name__ == "__main__":
    main()
