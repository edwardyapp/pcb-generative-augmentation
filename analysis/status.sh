#!/bin/bash
# One-glance status of the ICETA experiment.   ./analysis/status.sh    (or: watch -n 60 ./analysis/status.sh)
cd /mnt/storage/PycharmProjects/vq-vae-2-pytorch

bar () { local c=$1 t=$2 w=${3:-20}; [ "$t" -eq 0 ] && return
  local f=$(( c * w / t )); printf '['
  for ((i=0;i<w;i++)); do [ $i -lt $f ] && printf '#' || printf '.'; done
  printf ']'; }

echo "════════════════ ICETA experiment ════════════════"
date '+  %a %d %b %H:%M'

# ---------- phase 1: DONE ----------
echo
echo "✔ DONE"
echo "   crop-scale result ......... 600px 0.246 -> tight 0.896 (size-matched, 3 seeds)"
echo "   A/B/C recon ceiling ....... +0.51 @10% (no generator) vs own-budget recon +0.09  (fig10)"
echo "   conditioning trend ........ ep80 22.4% -> ep320 29.2%   GATE PASSED"
echo "   prior NLL ................. train 0.009 / held-out 5.12 -> overfitting (fig12)"
echo "   duplicate-image claim ..... RETRACTED (duplicates_review.html)"

# ---------- phase 2: budget-restricted generators ----------
echo
if pgrep -f "[r]un_budget_pipeline" > /dev/null; then   # matches the _resume script too
  CUR=$(grep -aoE "BUDGET [0-9]+%" logs/budget_pipeline.log | tail -1)
  STEP=$(grep -aoE "\[[0-9]+%\] [0-9]/6 [^|]{0,44}" logs/budget_pipeline.log | tail -1)
  echo "▶ RUNNING — budget-restricted generators   (${CUR:-starting})"
  [ -n "$STEP" ] && echo "   step: $STEP"
else
  echo "■ budget pipeline NOT running"
  grep -aE "FATAL|Traceback" logs/budget_pipeline.log 2>/dev/null | tail -2 | sed 's/^/   ! /'
fi

echo
printf "   %-6s %-12s %-22s %-22s %-12s %s\n" budget VQ-VAE "top prior" "bottom prior" samples A/B/C
for B in 10 100 25 50; do
  if [ "$B" = 100 ]; then VQ="reuses tight_560"; VN=14
  else VN=$(ls checkpoint/vqvae_b${B}_*.pt 2>/dev/null | wc -l); VQ="$(bar $VN 14) $VN/14"; fi
  T=$(ls checkpoint/prior_b${B}_top_*.pt    2>/dev/null | wc -l)
  BO=$(ls checkpoint/prior_b${B}_bottom_*.pt 2>/dev/null | wc -l)
  S=$(ls synth_b${B}/*.png 2>/dev/null | wc -l)
  A=$([ -f "results/abc_budget_b${B}.json" ] && echo "DONE" || echo "-")
  printf "   %-6s %-12s %s %-3s %s %-3s %s %3s/360  %s\n" \
    "b${B}%" "$(echo $VQ | cut -c1-12)" \
    "$(bar $T 8)" "$((T*40))" "$(bar $BO 8)" "$((BO*40))" "$(bar $S 360 8)" "$S" "$A"
done
echo "   (prior bars: epochs of 320; a checkpoint every 40)"

# ---------- results as they land ----------
echo
echo "── budget-restricted A/B/C (the honest curve) ──"
python - <<'PY' 2>/dev/null || echo "   none yet"
import json, os, numpy as np
any_=False
for b in (10,25,50,100):
    p=f'results/abc_budget_b{b}.json'
    if not os.path.exists(p): continue
    any_=True
    r=json.load(open(p))
    f=lambda k: np.array([x[k]['macro_f1'] for x in r])
    ka=np.mean([x['keep_rate'] for x in r])
    print(f"   b{b:<4}% A {f('A').mean():.3f}±{f('A').std():.3f}  "
          f"B {f('B').mean():.3f}±{f('B').std():.3f}  "
          f"C {f('C').mean():.3f}±{f('C').std():.3f}   "
          f"(B−A {f('B').mean()-f('A').mean():+.3f})  filter keeps {100*ka:.0f}%")
if not any_: raise SystemExit(1)
PY

echo
echo "── conditioning of each from-scratch budget generator ──"
if [ -f results/cond_checks_budget.jsonl ]; then
  python analysis/status_checks.py 2>/dev/null | grep budget || echo "   none yet"
else
  echo "   none yet"
fi

# ---------- machine ----------
echo
echo "── machine ──"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/   /'
df -h /mnt/storage | tail -1 | awk '{print "   disk /mnt/storage: "$4" free of "$2" ("$5" used)"}'
echo
echo "   live log:  tail -f logs/budget_pipeline.log"
echo "   pages:     review.html  generation_review.html  duplicates_review.html"
