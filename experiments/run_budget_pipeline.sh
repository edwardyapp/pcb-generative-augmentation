#!/bin/bash
# BUDGET-RESTRICTED GENERATORS -- the leakage-free experiment. Launch-gated.
#
# ORDER: 10 100 25 50.  The cut order is "50 first, then 25", so 10% and 100% are the
# protected pair and are built FIRST. If time runs out we still hold the two budgets that
# tell the story.
#
# WHY 100% IS RETRAINED FROM SCRATCH HERE, despite a 320-epoch run already existing:
#   pixelsnail_condtight3  <- condtight2_080  <- condtight_060  <- pixelsnail_top_357
#   and pixelsnail_top_357 was trained on lmdb/all == ALL 10 BOARDS, TEST BOARDS INCLUDED.
#   Through its warm-start the in-flight run has seen boards 06/09. That is harmless for the
#   GATE (the conditioning check never scores against test boards, and a leaked init can only
#   help conditioning -- so failing there is a STRONGER negative). It is NOT acceptable for an
#   A/B/C scored on boards 06/09: the synthetic data would descend from test-board weights.
#   So condtight3 = the gate; prior_b100 (from scratch, train-pool codes only) = the curve.
#
# At every budget EVERYTHING is from scratch on that budget's frozen subsample -- VQ-VAE too.
# A full-pool decoder has already learned to render defects from crops a budget-b model is not
# allowed to have seen, so reusing it would leak exactly the effect we are measuring.
# (100% is the sole exception by definition: its "subsample" IS the whole train pool, and
#  vqvae_tight_560 was itself trained from scratch on train-pool crops with test boards
#  excluded and asserted -- so it is clean.)
set -u
cd /mnt/storage/PycharmProjects/vq-vae-2-pytorch

echo "[wait] for the 320-epoch conditioning check..."
while ! grep -q '"epoch": 320' results/cond_checks.jsonl 2>/dev/null; do sleep 180; done
# the epoch-320 check finishes on one GPU while another check may still hold the other card;
# don't start training into a busy GPU
echo "[wait] for the remaining conditioning checks to release the GPUs..."
while pgrep -f "[c]ond_check.py" > /dev/null; do sleep 60; done

echo
python experiments/gate_check.py
GATE=$?
if [ "$GATE" -ne 0 ]; then
  echo
  echo "=============================================================="
  echo " GATE FAILED (or not ready: exit $GATE)."
  echo " NOT training budget-restricted generators."
  echo " If conditioning fails with the FULL train pool, a 320-epoch"
  echo " budget, AND a test-leaked warm start, it cannot work on 215"
  echo " crops from scratch. Shipping the negative result."
  echo "=============================================================="
  exit "$GATE"
fi

echo
echo "GATE PASSED -- training budget-restricted generators from scratch."

for B in 10 100 25 50; do
  echo ""
  echo "################ BUDGET ${B}% ################"

  if [ "$B" = "100" ]; then
    MAN="results/manifest_tight.csv"
    VQ="checkpoint/vqvae_tight_560.pt"          # already from-scratch, train-pool only
    LMDB="lmdb/train_pool_tight"                # codes already extracted with that VQ-VAE
    echo "[100%] VQ-VAE: reusing $VQ (from scratch, train pool only, test boards asserted out)"
  else
    MAN="results/manifest_tight_b${B}.csv"
    echo "[${B}%] 1/5 VQ-VAE from scratch on the budget-${B} subsample ONLY"
    CUDA_VISIBLE_DEVICES=1 python src/train_vqvae_tight.py \
      --manifest "$MAN" --split train --epoch 560 --batch 64 \
      --out_prefix "checkpoint/vqvae_b${B}"

    # HARD GATE. A previous run silently used a randomly-initialised VQ-VAE (the loader yielded
    # zero batches, so the model never took a gradient step) and built codes, priors, samples and
    # A/B/C on top of it. Every stage must now PROVE it produced what it claims.
    VQ="checkpoint/vqvae_b${B}_560.pt"
    if [ ! -f "$VQ" ]; then
      echo "FATAL [${B}%]: VQ-VAE did not reach epoch 560 ($VQ missing). Refusing to continue"
      echo "               with an undertrained decoder. Aborting."
      exit 1
    fi
    LMDB="lmdb/b${B}"
    echo "[${B}%] 2/5 extract codes with the budget-${B} VQ-VAE -> $VQ"
    rm -rf "$LMDB"
    CUDA_VISIBLE_DEVICES=1 python src/extract_code_labeled.py \
      --ckpt "$VQ" --manifest "$MAN" --split train --out "$LMDB"
  fi

  echo "[${B}%] 3/5 priors FROM SCRATCH (no --warm): top on 5090, bottom on 4090, in parallel"
  CUDA_VISIBLE_DEVICES=1 python src/train_pixelsnail_cond.py --hier top --epoch 320 \
    --path "$LMDB" --n_img_class 6 --out_prefix "checkpoint/prior_b${B}" &
  TOP_PID=$!
  CUDA_VISIBLE_DEVICES=0 python src/train_pixelsnail_cond.py --hier bottom --epoch 320 \
    --path "$LMDB" --n_img_class 6 --out_prefix "checkpoint/prior_b${B}" &
  BOT_PID=$!
  wait $TOP_PID $BOT_PID

  for P in "checkpoint/prior_b${B}_top_320.pt" "checkpoint/prior_b${B}_bottom_320.pt"; do
    if [ ! -f "$P" ]; then
      echo "FATAL [${B}%]: prior did not reach epoch 320 ($P missing). Aborting rather than"
      echo "               sampling from a half-trained prior."
      exit 1
    fi
  done
  AVAIL=$(df --output=avail -BG /mnt/storage | tail -1 | tr -dc '0-9')
  if [ "$AVAIL" -lt 20 ]; then
    echo "FATAL: only ${AVAIL}GB free. The last run died mid-write and corrupted checkpoints."
    exit 1
  fi

  # Conditioning check on THIS generator, which is clean (from scratch, no leaked warm-start).
  # The gate ran on condtight3, which is test-leaked via warm-start; that is the FRIENDLIEST
  # possible test and is sound as a gate. But no published conditioning number should rest on a
  # leaked model, so every from-scratch generator gets its own check. It also answers a question
  # worth a figure on its own: does conditioning DEGRADE as the generator's data shrinks?
  # The judge is held fixed (the 100% tight ResNet-18) across all budgets so the numbers are
  # comparable; it is a measuring instrument only and trains nothing.
  echo "[${B}%] 4/6 conditioning check on the from-scratch budget-${B} generator"
  CUDA_VISIBLE_DEVICES=1 python experiments/cond_check.py \
    --vqvae "$VQ" \
    --top    "checkpoint/prior_b${B}_top_320.pt" \
    --bottom "checkpoint/prior_b${B}_bottom_320.pt" \
    --epoch 320 --tag "prior_b${B}" --per_class 12 --chunk 36 \
    --out results/cond_checks_budget.jsonl

  echo "[${B}%] 5/6 sample a labelled synthetic pool (60/class = 360)"
  CUDA_VISIBLE_DEVICES=1 python experiments/sample_pool.py \
    --vqvae "$VQ" \
    --top    "checkpoint/prior_b${B}_top_320.pt" \
    --bottom "checkpoint/prior_b${B}_bottom_320.pt" \
    --out    "synth_b${B}" --per_class 60 --chunk 36

  echo "[${B}%] 6/6 A/B/C  (filter = the budget-${B} Condition-A model, SAME seed)"
  CUDA_VISIBLE_DEVICES=0 python experiments/abc_budget.py --budget "$B" --synth "synth_b${B}" \
    --manifest "$MAN"

  echo "################ BUDGET ${B}% DONE ################"
done

python analysis/plot_budget_abc.py || true
echo "===== BUDGET-RESTRICTED PIPELINE COMPLETE ====="
