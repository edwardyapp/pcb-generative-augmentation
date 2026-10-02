"""Pre-generate a pool of PCB-defect samples for the live demo.

Loads the trained VQ-VAE-2 and the top/bottom PixelSNAIL priors, samples top
codes then bottom codes autoregressively (adapting the sampling logic in
sample.py), decodes to images and saves them as individual 512x512 PNGs.

The heavy lifting (model loading + sampling) is exposed as importable helpers
so demo_app.py can reuse it for optional live sampling.

Example:
    python legacy/generate_pool.py --n 300 --batch 16 --temp 1.0 --out demo_pool/
"""

import sys, pathlib  # repo layout (README.md): src/, experiments/, analysis/ importable from anywhere
sys.path[1:1] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('src', 'experiments', 'analysis')]
import argparse
import os
import time

import torch
from PIL import Image

from vqvae import VQVAE
from pixelsnail import PixelSNAIL

# Model args inferred from the training scripts and confirmed against the
# checkpoint state-dict shapes:
#   - VQVAE(): rosinality defaults (channel=128, embed_dim=64, n_embed=512)
#   - PixelSNAIL: channel=256, n_res_channel=256 for both levels (from ckpt args)
# For 256x256 inputs the latents are top=32x32, bottom=64x64.
TOP_SHAPE = [32, 32]
BOTTOM_SHAPE = [64, 64]
N_CLASS = 512

DEFAULT_CKPT_DIR = "checkpoint"
DEFAULT_VQVAE = "vqvae_560.pt"
DEFAULT_TOP = "pixelsnail_top_357.pt"
DEFAULT_BOTTOM = "pixelsnail_bottom_best.pt"


@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None):
    """Autoregressively sample a grid of discrete codes (from sample.py)."""
    row = torch.zeros(batch, *size, dtype=torch.int64, device=device)
    cache = {}

    for i in range(size[0]):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row


def _load_pixelsnail(hier, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt["args"]

    if hier == "top":
        model = PixelSNAIL(
            TOP_SHAPE,
            N_CLASS,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
        )
    elif hier == "bottom":
        model = PixelSNAIL(
            BOTTOM_SHAPE,
            N_CLASS,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
        )
    else:
        raise ValueError(hier)

    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def load_models(
    ckpt_dir=DEFAULT_CKPT_DIR,
    vqvae_name=DEFAULT_VQVAE,
    top_name=DEFAULT_TOP,
    bottom_name=DEFAULT_BOTTOM,
    device="cuda",
):
    """Load VQ-VAE-2 + both PixelSNAIL priors onto `device`. Returns a dict."""
    vqvae_path = os.path.join(ckpt_dir, vqvae_name)
    top_path = os.path.join(ckpt_dir, top_name)
    bottom_path = os.path.join(ckpt_dir, bottom_name)
    for p in (vqvae_path, top_path, bottom_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"checkpoint not found: {p}")

    vqvae = VQVAE()
    vqvae.load_state_dict(
        torch.load(vqvae_path, map_location=device, weights_only=False)
    )
    vqvae = vqvae.to(device).eval()

    top = _load_pixelsnail("top", top_path, device)
    bottom = _load_pixelsnail("bottom", bottom_path, device)

    return {"vqvae": vqvae, "top": top, "bottom": bottom, "device": device}


@torch.no_grad()
def sample_decoded(models, batch, temperature):
    """Sample `batch` images end-to-end. Returns a (batch,3,256,256) tensor in [-1,1]."""
    device = models["device"]
    top_sample = sample_model(models["top"], device, batch, TOP_SHAPE, temperature)
    bottom_sample = sample_model(
        models["bottom"], device, batch, BOTTOM_SHAPE, temperature, condition=top_sample
    )
    decoded = models["vqvae"].decode_code(top_sample, bottom_sample)
    return decoded.clamp(-1, 1)


def tensor_to_pil(img):
    """(3,H,W) float tensor in [-1,1] -> PIL RGB image."""
    img = (img.clamp(-1, 1) + 1) / 2
    img = (img * 255 + 0.5).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(img.permute(1, 2, 0).cpu().numpy())


def upscale(pil_img, size=512):
    """Lanczos-upscale a PIL image to size x size."""
    if pil_img.size == (size, size):
        return pil_img
    return pil_img.resize((size, size), Image.LANCZOS)


def _fmt_eta(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300, help="total images in the pool")
    parser.add_argument("--batch", type=int, default=16, help="batch size (auto-reduces on OOM)")
    parser.add_argument("--temp", type=float, default=1.0, help="sampling temperature")
    parser.add_argument("--out", type=str, default="demo_pool/", help="output directory")
    parser.add_argument("--size", type=int, default=512, help="output image size (lanczos upscale)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ckpt-dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--vqvae", type=str, default=DEFAULT_VQVAE)
    parser.add_argument("--top", type=str, default=DEFAULT_TOP)
    parser.add_argument("--bottom", type=str, default=DEFAULT_BOTTOM)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    os.makedirs(args.out, exist_ok=True)

    # Stable zero-padded filenames so a test run and the full run share the pool.
    width = max(4, len(str(args.n)))

    def path_for(idx):
        return os.path.join(args.out, f"sample_{idx:0{width}d}.png")

    missing = [i for i in range(args.n) if not os.path.exists(path_for(i))]
    already = args.n - len(missing)

    print(f"Loading models on {args.device} ...")
    models = load_models(
        ckpt_dir=args.ckpt_dir,
        vqvae_name=args.vqvae,
        top_name=args.top,
        bottom_name=args.bottom,
        device=args.device,
    )
    if args.device.startswith("cuda"):
        name = torch.cuda.get_device_name(models["device"])
        print(f"Models loaded on {models['device']} ({name}).")
    else:
        print("Models loaded.")

    print(
        f"Pool target: {args.n} | already present: {already} | to generate: {len(missing)}"
    )
    if not missing:
        print(f"Nothing to do -- pool already complete in {args.out}")
        return

    batch = max(1, args.batch)
    done = 0
    start = time.time()
    ptr = 0

    while ptr < len(missing):
        bs = min(batch, len(missing) - ptr)
        chunk = missing[ptr : ptr + bs]

        try:
            decoded = sample_decoded(models, bs, args.temp)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() and batch > 1:
                torch.cuda.empty_cache()
                new_batch = max(1, batch // 2)
                print(f"\n[OOM] reducing batch {batch} -> {new_batch} and retrying")
                batch = new_batch
                continue  # retry the same chunk with a smaller batch
            raise

        for k, gidx in enumerate(chunk):
            img = upscale(tensor_to_pil(decoded[k]), args.size)
            img.save(path_for(gidx))

        ptr += bs
        done += bs

        elapsed = time.time() - start
        rate = done / elapsed  # images/sec
        remaining = len(missing) - done
        eta = remaining / rate if rate > 0 else 0
        print(
            f"  {already + done}/{args.n} saved "
            f"| {rate:.3f} img/s ({1 / rate:.1f} s/img) "
            f"| elapsed {_fmt_eta(elapsed)} | ETA {_fmt_eta(eta)}",
            flush=True,
        )

    total = time.time() - start
    print(
        f"\nDone. Generated {done} images in {_fmt_eta(total)} "
        f"({total / done:.1f} s/img). Pool complete: {args.out}"
    )


if __name__ == "__main__":
    main()
