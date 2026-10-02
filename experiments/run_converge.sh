#!/bin/bash
# Continue the tight conditional priors from epoch 80 -> +240 epochs (=320 total, close to
# the original run's 357). Loss was still falling at 80; the conditioning check was premature.
# New prefix condtight3 => the condtight2 checkpoints are NOT overwritten. Ckpt every epoch,
# so this can be killed at any point and the best available epoch still used.
set -e
cd /mnt/storage/PycharmProjects/vq-vae-2-pytorch
while pgrep -f "[s]ample_uncond.py" > /dev/null; do sleep 60; done   # wait for the GPU
echo "===== [1/2] TOP prior, continue 80 -> 320 ====="
CUDA_VISIBLE_DEVICES=1 python src/train_pixelsnail_cond.py --hier top --epoch 240 \
  --warm checkpoint/pixelsnail_condtight2_top_080.pt \
  --path lmdb/train_pool_tight --n_img_class 6 \
  --out_prefix checkpoint/pixelsnail_condtight3
echo "===== [2/2] BOTTOM prior, continue 80 -> 320 ====="
CUDA_VISIBLE_DEVICES=1 python src/train_pixelsnail_cond.py --hier bottom --epoch 240 \
  --warm checkpoint/pixelsnail_condtight2_bottom_080.pt \
  --path lmdb/train_pool_tight --n_img_class 6 \
  --out_prefix checkpoint/pixelsnail_condtight3
echo "===== CONVERGE RUN DONE ====="
