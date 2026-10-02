#!/bin/bash
# RESUME of run_budget_pipeline.sh after the 2026-07-13 13:31 FATAL.
#
# What happened: the b100 bottom prior OOM'd at launch (09:54) because an n=360
# conditioning-recheck chunk was still holding ~13GB on GPU 0 — the recheck's stage gate
# assumed abc_budget would outlast a 12-min chunk, but at b10 abc_budget finishes in
# minutes. The top prior trained to 320 unharmed (checkpoint/prior_b100_top_320.pt); the
# pipeline then hit its own missing-checkpoint gate and aborted, exactly as designed.
#
# This script is the VERBATIM remaining tail of the original (same commands, same GPU
# assignments, same hard gates), plus idempotence: every stage is skipped if its output
# already exists, so this script can be re-run safely after any interruption.
# b10 is complete (results/abc_budget_b10.json) and is skipped by the same rule.
set -u
cd /mnt/storage/PycharmProjects/vq-vae-2-pytorch

for B in 10 100 25 50; do
  echo ""
  echo "################ BUDGET ${B}% ################"
  if [ -f "results/abc_budget_b${B}.json" ]; then
    echo "[${B}%] abc_budget_b${B}.json exists — budget complete, skipping"
    continue
  fi

  if [ "$B" = "100" ]; then
    MAN="results/manifest_tight.csv"
    VQ="checkpoint/vqvae_tight_560.pt"
    LMDB="lmdb/train_pool_tight"
    echo "[100%] VQ-VAE: reusing $VQ (from scratch, train pool only, test boards asserted out)"
  else
    MAN="results/manifest_tight_b${B}.csv"
    VQ="checkpoint/vqvae_b${B}_560.pt"
    LMDB="lmdb/b${B}"
    if [ -f "checkpoint/prior_b${B}_top_320.pt" ] && [ -f "checkpoint/prior_b${B}_bottom_320.pt" ]; then
      echo "[${B}%] priors already exist — skipping VQ-VAE/extract/prior training"
    else
      if [ ! -f "$VQ" ]; then
        echo "[${B}%] 1/5 VQ-VAE from scratch on the budget-${B} subsample ONLY"
        CUDA_VISIBLE_DEVICES=1 python src/train_vqvae_tight.py \
          --manifest "$MAN" --split train --epoch 560 --batch 64 \
          --out_prefix "checkpoint/vqvae_b${B}"
        if [ ! -f "$VQ" ]; then
          echo "FATAL [${B}%]: VQ-VAE did not reach epoch 560 ($VQ missing). Aborting."
          exit 1
        fi
      fi
      echo "[${B}%] 2/5 extract codes with the budget-${B} VQ-VAE -> $VQ"
      rm -rf "$LMDB"
      CUDA_VISIBLE_DEVICES=1 python src/extract_code_labeled.py \
        --ckpt "$VQ" --manifest "$MAN" --split train --out "$LMDB"
    fi
  fi

  if [ ! -f "checkpoint/prior_b${B}_top_320.pt" ] || [ ! -f "checkpoint/prior_b${B}_bottom_320.pt" ]; then
    echo "[${B}%] 3/5 priors FROM SCRATCH: top on 5090, bottom on 4090, in parallel"
    TOP_PID=""; BOT_PID=""
    if [ ! -f "checkpoint/prior_b${B}_top_320.pt" ]; then
      CUDA_VISIBLE_DEVICES=1 python src/train_pixelsnail_cond.py --hier top --epoch 320 \
        --path "$LMDB" --n_img_class 6 --out_prefix "checkpoint/prior_b${B}" &
      TOP_PID=$!
    fi
    if [ ! -f "checkpoint/prior_b${B}_bottom_320.pt" ]; then
      CUDA_VISIBLE_DEVICES=0 python src/train_pixelsnail_cond.py --hier bottom --epoch 320 \
        --path "$LMDB" --n_img_class 6 --out_prefix "checkpoint/prior_b${B}" &
      BOT_PID=$!
    fi
    wait $TOP_PID $BOT_PID
  fi

  for P in "checkpoint/prior_b${B}_top_320.pt" "checkpoint/prior_b${B}_bottom_320.pt"; do
    if [ ! -f "$P" ]; then
      echo "FATAL [${B}%]: prior did not reach epoch 320 ($P missing). Aborting rather than"
      echo "               sampling from a half-trained prior."
      exit 1
    fi
  done
  AVAIL=$(df --output=avail -BG /mnt/storage | tail -1 | tr -dc '0-9')
  if [ "$AVAIL" -lt 20 ]; then
    echo "FATAL: only ${AVAIL}GB free. Refusing to risk corrupt checkpoints."
    exit 1
  fi

  if ! grep -q "\"tag\": \"prior_b${B}\"" results/cond_checks_budget.jsonl 2>/dev/null; then
    echo "[${B}%] 4/6 conditioning check on the from-scratch budget-${B} generator"
    CUDA_VISIBLE_DEVICES=1 python experiments/cond_check.py \
      --vqvae "$VQ" \
      --top    "checkpoint/prior_b${B}_top_320.pt" \
      --bottom "checkpoint/prior_b${B}_bottom_320.pt" \
      --epoch 320 --tag "prior_b${B}" --per_class 12 --chunk 36 \
      --out results/cond_checks_budget.jsonl
  fi

  if [ "$(ls "synth_b${B}" 2>/dev/null | wc -l)" -lt 360 ]; then
    echo "[${B}%] 5/6 sample a labelled synthetic pool (60/class = 360)"
    CUDA_VISIBLE_DEVICES=1 python experiments/sample_pool.py \
      --vqvae "$VQ" \
      --top    "checkpoint/prior_b${B}_top_320.pt" \
      --bottom "checkpoint/prior_b${B}_bottom_320.pt" \
      --out    "synth_b${B}" --per_class 60 --chunk 36
  fi

  echo "[${B}%] 6/6 A/B/C  (filter = the budget-${B} Condition-A model, SAME seed)"
  CUDA_VISIBLE_DEVICES=0 python experiments/abc_budget.py --budget "$B" --synth "synth_b${B}" \
    --manifest "$MAN"

  echo "################ BUDGET ${B}% DONE ################"
done

python analysis/plot_budget_abc.py || true
echo "===== BUDGET-RESTRICTED PIPELINE COMPLETE ====="
