#!/usr/bin/env bash
set -e
cd C:/hack
export PYTHONUNBUFFERED=1

for SEED in 2026 1337 4242; do
  OUT="artifacts/models/boxing_tcn_v2_s${SEED}.pt"
  if [ -f "$OUT" ]; then
    echo "=== [$(date)] skip seed $SEED (exists) ==="
    continue
  fi
  echo "=== [$(date)] TRAIN seed=$SEED ==="
  python -u scripts/train_tcn_v2.py --out "$OUT" --seed $SEED --epochs 18
done

echo "=== [$(date)] TUNE thresholds ==="
python -u scripts/tune_threshold_v2.py --ckpts \
  artifacts/models/boxing_tcn_v2_s2026.pt \
  artifacts/models/boxing_tcn_v2_s1337.pt \
  artifacts/models/boxing_tcn_v2_s4242.pt \
  > artifacts/tune_v2.log

echo "=== [$(date)] PREDICT ensemble ==="
THR=$(grep BEST artifacts/tune_v2.log | awk '{print $2}' | sed 's/threshold=//')
MD=$(grep BEST artifacts/tune_v2.log | awk '{print $3}' | sed 's/min_distance=//')
echo "Using threshold=$THR min_distance=$MD"
python -u scripts/predict_tcn_v2.py \
  --ckpts artifacts/models/boxing_tcn_v2_s2026.pt artifacts/models/boxing_tcn_v2_s1337.pt artifacts/models/boxing_tcn_v2_s4242.pt \
  --threshold ${THR:-0.55} --min_distance ${MD:-8} \
  --out submission_tcn_v3.csv

echo "=== [$(date)] DONE v2 ==="
