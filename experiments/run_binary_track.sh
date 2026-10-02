#!/bin/bash
# BINARY TRACK — generator + A/B/C. Queued behind the budget pipeline (both GPUs are needed
# during its prior-training phases, so there is no persistently free card to steal).
#
# THE LEAKAGE PROBLEM, stated before we spend the GPU-hours:
#   "Retrain the unconditional prior on train-pool tight crops" means the generator sees ALL
#   4,274 train crops. Sampling from it and adding those samples to a 10% BUDGET hands the
#   classifier back information from the other 90% -- the exact effect abc_recon.py already
#   measured (B_pool @10% = 0.910 vs a full-data ceiling of 0.918: +0.51 of pure leakage).
#   So a full-pool generator gives the LEAKY number, not the honest one.
# Therefore we train BOTH:
#   uncond_tight        - on the whole train pool  -> the LEAKY curve (what the field reports)
#   uncond_b{10,25,50}  - on the frozen budget subsample only -> the HONEST curve
# and report them side by side, exactly as with the class-conditional track.
#
# OUTCOME (added 2026-10-02): the hypothesis above did not hold. +0.51 is a reconstruction
#   ceiling (abc_recon.py uses no generator), not what a full-pool generator delivers. The
#   full-pool generator gives +0.002 at 10% here and in the 6-way track (AUDIT.md section 7).
set -u
cd /mnt/storage/PycharmProjects/vq-vae-2-pytorch

echo "[wait] for the budget pipeline (original OR resume) to release the GPUs..."
# no .sh anchor: must also match run_budget_pipeline_resume.sh. And double-check no
# prior trainer is mid-flight (2026-07-13: this script fired in the 90s gap after the
# pipeline FATAL'd and stomped both GPUs).
while pgrep -f "[r]un_budget_pipeline" > /dev/null || \
      pgrep -f "[t]rain_pixelsnail_cond.py" > /dev/null; do sleep 300; done
echo "[go  ] GPUs free"

train_uncond () {   # $1 = tag, $2 = lmdb path
  local TAG=$1 LMDB=$2
  echo "=== unconditional priors (n_img_class=0, NO modifications) on $LMDB -> $TAG ==="
  CUDA_VISIBLE_DEVICES=1 python src/train_pixelsnail_cond.py --hier top --epoch 320 \
    --path "$LMDB" --n_img_class 0 --out_prefix "checkpoint/${TAG}" &
  local A=$!
  CUDA_VISIBLE_DEVICES=0 python src/train_pixelsnail_cond.py --hier bottom --epoch 320 \
    --path "$LMDB" --n_img_class 0 --out_prefix "checkpoint/${TAG}" &
  local B=$!
  wait $A $B
  for P in "checkpoint/${TAG}_top_320.pt" "checkpoint/${TAG}_bottom_320.pt"; do
    [ -f "$P" ] || { echo "FATAL: $P missing"; exit 1; }
  done
}

# ---- LEAKY generator: the whole train pool ----
train_uncond uncond_tight lmdb/train_pool_tight
CUDA_VISIBLE_DEVICES=1 python experiments/sample_pool.py --vqvae checkpoint/vqvae_tight_560.pt \
  --top checkpoint/uncond_tight_top_320.pt --bottom checkpoint/uncond_tight_bottom_320.pt \
  --out synth_binary_pool --per_class 60 --chunk 36 --uncond

# ---- HONEST generators: budget-restricted, reusing each budget's from-scratch VQ-VAE ----
for B in 10 25 50; do
  VQ="checkpoint/vqvae_b${B}_560.pt"
  [ -f "$VQ" ] || { echo "SKIP b${B}: $VQ not built yet"; continue; }
  train_uncond "uncond_b${B}" "lmdb/b${B}"
  CUDA_VISIBLE_DEVICES=1 python experiments/sample_pool.py --vqvae "$VQ" \
    --top "checkpoint/uncond_b${B}_top_320.pt" --bottom "checkpoint/uncond_b${B}_bottom_320.pt" \
    --out "synth_binary_b${B}" --per_class 60 --chunk 36 --uncond
done

echo "=== binary A/B/C ==="
CUDA_VISIBLE_DEVICES=0 python experiments/abc_binary.py
echo "===== BINARY TRACK DONE ====="
