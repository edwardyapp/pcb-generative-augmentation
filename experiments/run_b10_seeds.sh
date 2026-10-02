#!/bin/bash
# 10-seed b10 A/B/C on the idle 5090, with a watchdog that YIELDS the card the moment the
# budget pipeline wants it. Results are written per seed, so a kill costs at most one seed.
#
# The pipeline claims GPU 1 for b100's conditioning check the instant the bottom prior finishes.
# We watch for that and kill ourselves first -- politely, rather than causing the second OOM.
set -u
cd /mnt/storage/PycharmProjects/vq-vae-2-pytorch

CUDA_VISIBLE_DEVICES=1 python experiments/abc_b10_seeds.py --seeds 10 &
JOB=$!
echo "b10 10-seed A/B/C started on GPU1 (pid $JOB)"

while kill -0 "$JOB" 2>/dev/null; do
  # the pipeline's next GPU1 stage is cond_check / sample_pool / train top prior
  if pgrep -f "[c]ond_check.py|[s]ample_pool.py" > /dev/null || \
     pgrep -f "[t]rain_pixelsnail_cond.py --hier top" > /dev/null; then
    echo "!! budget pipeline wants GPU1 — yielding now (partial results are already on disk)"
    kill "$JOB" 2>/dev/null
    sleep 5
    kill -9 "$JOB" 2>/dev/null
    echo "yielded. Re-run  python experiments/abc_b10_seeds.py --seeds 10  later to finish the seeds."
    exit 0
  fi
  sleep 20
done
wait "$JOB" || true
echo "===== b10 10-SEED A/B/C DONE ====="
