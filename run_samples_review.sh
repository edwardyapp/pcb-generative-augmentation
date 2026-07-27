#!/bin/bash
# Build samples_review.html once the b25 binary sampler finishes (so its pool is complete).
# Pinned to GPU1: during b50 prior training the 5090 keeps ~15GB free; this build needs ~4GB.
set -u
cd /mnt/storage/PycharmProjects/vq-vae-2-pytorch
while pgrep -f "[s]ample_pool.py" > /dev/null; do sleep 60; done
CUDA_VISIBLE_DEVICES=1 python build_samples_review.py
