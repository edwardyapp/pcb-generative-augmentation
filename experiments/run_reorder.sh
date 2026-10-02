#!/bin/bash
# REORDER: binary track BEFORE b25/b50.
#
# The running run_budget_pipeline_resume.sh loops `for B in 10 100 25 50`. We cannot edit it --
# bash reads a script incrementally as it executes, so editing it mid-run can make it run garbage.
# Instead we let it finish b100 and then stop it at a SAFE BOUNDARY: the moment b100's A/B/C JSON
# lands (its last step), before it can start b25's VQ-VAE.
#
# Rationale (the user's): binary is an INDEPENDENT test of the paper's central claim. b25/b50 are
# interior points on a curve whose endpoints (b10, b100) we already have.
#
# Order enforced:  ...b100  ->  binary track  ->  b25, b50
set -u
cd /mnt/storage/PycharmProjects/vq-vae-2-pytorch

echo "[wait] for b100's A/B/C to land (the resume script's last b100 step)..."
while [ ! -f results/abc_budget_b100.json ]; do sleep 60; done
echo "[go  ] b100 complete."

# stop the driver before it starts b25; leave any b100 python alone (it is finished by now)
sleep 10
PID=$(pgrep -f "[r]un_budget_pipeline_resume.sh" | head -1)
if [ -n "$PID" ]; then
  kill "$PID" 2>/dev/null
  echo "stopped the budget driver (pid $PID) at the b100/b25 boundary"
fi
# if it had already begun b25's VQ-VAE, stop that too (cheap to redo, ~10 min)
sleep 3
pkill -f "[t]rain_vqvae_tight.py --manifest results/manifest_tight_b25" 2>/dev/null && \
  echo "stopped a just-started b25 VQ-VAE (it will be redone later)"

echo "[go  ] run_binary_track.sh is already armed and waiting; it will now proceed."
while pgrep -f "[r]un_binary_track.sh" > /dev/null; do sleep 120; done
echo "[done] binary track finished."

echo "[go  ] now the interior points: b25, b50"
setsid nohup ./experiments/run_budget_b25_b50.sh > logs/budget_b25_b50.log 2>&1 < /dev/null &
echo "===== REORDER COMPLETE — b25/b50 launched ====="
