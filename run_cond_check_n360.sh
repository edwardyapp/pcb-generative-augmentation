#!/bin/bash
# n>=360 re-check of the epoch-320 conditioning gate (audit action 1) — v2.
#
# v1 caused the 13:31 pipeline FATAL: it treated "sample_pool running" as a safe window,
# but a 12-min chunk launched near the END of sampling outlived the (minutes-short at low
# budget) abc_budget stage and was still holding GPU 0 when the next prior stage tried to
# allocate. v2 rules, checked before EVERY chunk:
#   GPU1 mode: the b100 bottom prior is training on GPU 0 (a >=13h stage of the sequential
#              resume script, during which nothing else can touch GPU 1) and GPU 1 is idle
#              -> run the chunk on GPU 1.
#   GPU0 mode: no bottom-hier trainer anywhere, GPU 0 idle, AND the driver is inside
#              cond_check (always followed by an ~88-min sample_pool) or inside sample_pool
#              with >=80 images still to write (~20 min > chunk length, measured from the
#              sampler's own --out dir) -> run on GPU 0.
#   Fallback:  no driver scripts and no trainers at all, GPU 0 idle -> run on GPU 0.
# Anything else: sleep. A chunk is 24 samples (4/class), ~10-12 min, ~13GB.
set -u
cd /mnt/storage/PycharmProjects/vq-vae-2-pytorch
OUT=results/cond_check_ep320_n360.jsonl
TAG=condtight3_n360
NEED=15

count() { c=$(grep -c "\"tag\": \"$TAG\"" "$OUT" 2>/dev/null); echo "${c:-0}"; }

run_chunk() {  # $1 = CUDA device index
  echo "[chunk $(($(count)+1))/$NEED] $(date '+%F %T')  on GPU $1"
  CUDA_VISIBLE_DEVICES=$1 python cond_check.py \
    --top    checkpoint/pixelsnail_condtight3_top_240.pt \
    --bottom checkpoint/pixelsnail_condtight3_bottom_240.pt \
    --epoch 320 --tag "$TAG" --per_class 4 --chunk 24 --out "$OUT"
}

while true; do
  N=$(count)
  [ "$N" -ge "$NEED" ] && break

  B100BOT=$(pgrep -cf '[t]rain_pixelsnail_cond.py.*hier bottom.*prior_b100' || true)
  ANYBOT=$(pgrep -cf '[t]rain_pixelsnail_cond.py.*hier bottom' || true)
  ANYTRAIN=$(pgrep -cf '[t]rain_pixelsnail_cond.py|[t]rain_vqvae' || true)
  DRIVER=$(pgrep -cf '[r]un_budget_pipeline|[r]un_binary_track' || true)
  USED0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  USED1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)

  if [ "${B100BOT:-0}" -gt 0 ] && [ "$USED1" -lt 3000 ]; then
    run_chunk 1
    continue
  fi

  if [ "${ANYBOT:-0}" -eq 0 ] && [ "$USED0" -lt 3000 ]; then
    if pgrep -f '[c]ond_check.py' > /dev/null; then
      run_chunk 0
      continue
    fi
    SP_CMD=$(pgrep -af '[s]ample_pool\.py' | head -1 || true)
    if [ -n "$SP_CMD" ]; then
      SP_OUT=$(echo "$SP_CMD" | grep -oE '\-\-out +[^ ]+' | awk '{print $2}')
      SP_PC=$(echo "$SP_CMD" | grep -oE '\-\-per_class +[0-9]+' | awk '{print $2}')
      SP_DONE=$(ls "${SP_OUT:-/nonexistent}" 2>/dev/null | wc -l)
      SP_REMAIN=$(( ${SP_PC:-60} * 6 - SP_DONE ))
      if [ "$SP_REMAIN" -ge 80 ]; then
        run_chunk 0
        continue
      fi
    elif [ "${DRIVER:-0}" -eq 0 ] && [ "${ANYTRAIN:-0}" -eq 0 ]; then
      run_chunk 0
      continue
    fi
  fi
  sleep 120
done

echo "==== all $NEED chunks done, pooling ===="
python cond_check_n360_report.py
echo "ALL_DONE"
