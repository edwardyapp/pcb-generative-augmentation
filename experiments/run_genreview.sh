#!/bin/bash
# Build generation_review.html. Waits for the running jobs; never elbows them off a GPU.
#   1. wait for prior training to finish (frees the 4090)
#   2. wait for the NLL eval already queued on that card
#   3. train the defect/no-defect detector  (~5 min)
#   4. wait for the epoch-320 conditioning check (section 3 uses its samples)
#   5. build the page
set -u
cd /mnt/storage/PycharmProjects/vq-vae-2-pytorch

echo "[wait] prior training..."
while pgrep -f "[t]rain_pixelsnail_cond.py" > /dev/null; do sleep 60; done
echo "[wait] NLL eval..."
while pgrep -f "[e]val_prior_nll.py" > /dev/null; do sleep 30; done

echo "[run ] defect/no-defect detector"
CUDA_VISIBLE_DEVICES=0 python experiments/make_defect_detector.py

echo "[wait] epoch-320 conditioning check (section 3 needs its samples)..."
while [ ! -f synth_trend/ep320_chunk1.png ]; do sleep 60; done
while pgrep -f "[c]ond_check.py" > /dev/null; do sleep 30; done

echo "[run ] building generation_review.html"
python analysis/build_generation_review.py --epoch 320

echo "===== GENERATION REVIEW BUILT ====="
