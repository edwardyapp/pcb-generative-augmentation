#!/bin/bash
# Conditioning checks at ABSOLUTE epochs 120..320  (condtight3 local epoch = absolute - 80).
#
# Runs OPPORTUNISTICALLY on the 5090: the top prior finished first, so that card is idle while
# the bottom prior grinds on the 4090. Each check only needs top_L and bottom_L to exist, and
# every epoch is checkpointed -- so a check for epoch 120 run now is identical to one run later.
# No reason to let a GPU idle for three hours waiting for training to end.
#
# GPU1 (5090, 32GB) = checks.  GPU0 (4090) = bottom prior training. They do not contend.
set -u
cd /mnt/storage/PycharmProjects/vq-vae-2-pytorch

for ABS in 120 160 200 240 280 320; do
  L=$(( ABS - 80 ))
  T=$(printf "checkpoint/pixelsnail_condtight3_top_%03d.pt" $L)
  B=$(printf "checkpoint/pixelsnail_condtight3_bottom_%03d.pt" $L)

  echo "[wait] absolute epoch $ABS (local $L) -> needs $(basename $T) + $(basename $B)"
  while [ ! -f "$T" ] || [ ! -f "$B" ]; do sleep 60; done
  sleep 15   # let the checkpoint write settle

  echo "[gpu1] conditioning check @ absolute epoch $ABS"
  CUDA_VISIBLE_DEVICES=1 python experiments/cond_check.py --top "$T" --bottom "$B" \
    --epoch "$ABS" --tag condtight3 --per_class 12 --chunk 36 --grid_dir synth_trend

  python analysis/plot_consistency.py || true
done

echo "===== ALL CONDITIONING CHECKS DONE ====="
