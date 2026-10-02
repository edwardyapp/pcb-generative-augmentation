#!/bin/bash
# Post-hoc train vs held-out NLL. Touches NOTHING that is running: waits for the bottom prior
# to finish, then uses the 4090 (which that frees) while conditioning checks continue on the 5090.
# Trains nothing; forward passes only.
set -u
cd /mnt/storage/PycharmProjects/vq-vae-2-pytorch

echo "[wait] for the priors to finish training..."
while pgrep -f "[t]rain_pixelsnail_cond.py" > /dev/null; do sleep 60; done
echo "[go  ] priors done"

# Encode the TEST-board tight crops into codes. The VQ-VAE (vqvae_tight_560) was trained on the
# train pool only with test boards asserted out, so this is a genuine held-out set for the priors.
# Encoding for evaluation is not training and leaks nothing back into any model.
if [ ! -d lmdb/test_tight ]; then
  echo "[prep] encoding test-board tight crops -> lmdb/test_tight"
  CUDA_VISIBLE_DEVICES=0 python src/extract_code_labeled.py \
    --ckpt checkpoint/vqvae_tight_560.pt \
    --manifest results/manifest_tight_perbbox.csv \
    --split test --out lmdb/test_tight
fi

echo "[run ] NLL across every checkpoint (80,120,...,320)"
CUDA_VISIBLE_DEVICES=0 python experiments/eval_prior_nll.py

python analysis/plot_nll.py || true
echo "===== NLL EVAL DONE ====="
