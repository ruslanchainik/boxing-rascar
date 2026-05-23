#!/usr/bin/env bash
set -e
cd C:/hack
export PYTHONUNBUFFERED=1

# Best postprocess so far (from LB ablations):
# - no snap-to-velocity (it hurts)
# - fighter smoothing on (default)
# - attr averaging with half-window 4 (window 9)
# - hand prior on
POST_FLAGS="--no_snap --hand_prior --attr_avg_half 4 --threshold 0.65 --min_distance 6"

# --------- STAGE A: 5-seed v1-style ensemble ----------
echo "=== [$(date)] STAGE A: 5-seed v1-style ==="
SEEDS_A="2026 1337 4242 31337 8088"
for s in $SEEDS_A; do
  OUT="artifacts/models/boxing_tcn_v1_s${s}.pt"
  if [ -f "$OUT" ]; then
    echo "skip seed $s (exists)"
    continue
  fi
  echo "=== [$(date)] train v1 seed=$s ==="
  python -u scripts/train_tcn.py --out "$OUT" --seed $s --epochs 30 \
    --hidden_dim 192 --batch_size 8 --crop_len 512 --crops_per_video 12 || true
done

echo "=== [$(date)] PREDICT ensemble v1-5seed ==="
python -u scripts/predict_tcn_v2.py \
  --ckpts artifacts/models/boxing_tcn_v1_s2026.pt \
          artifacts/models/boxing_tcn_v1_s1337.pt \
          artifacts/models/boxing_tcn_v1_s4242.pt \
          artifacts/models/boxing_tcn_v1_s31337.pt \
          artifacts/models/boxing_tcn_v1_s8088.pt \
  $POST_FLAGS \
  --out submission_v1_5seed.csv || true

# --------- STAGE B: V2 retrain v2 (fixed hyperparams) ----------
echo "=== [$(date)] STAGE B: V2 retrain v2 (dropout=0.2, no mixup, 30 epochs) ==="
SEEDS_B="2026 1337 4242"
for s in $SEEDS_B; do
  OUT="artifacts/models/boxing_tcn_v2fix_s${s}.pt"
  if [ -f "$OUT" ]; then
    echo "skip seed $s (exists)"
    continue
  fi
  echo "=== [$(date)] train v2fix seed=$s ==="
  python -u scripts/train_tcn_v2.py --out "$OUT" --seed $s --epochs 30 \
    --dropout 0.2 --weight_decay 1e-4 --mixup_prob 0.0 --color_swap_prob 0.5 \
    --label_smoothing 0.05 --ema_decay 0.995 --crops_per_video 12 || true
done

echo "=== [$(date)] PREDICT ensemble v2fix-3seed ==="
python -u scripts/predict_tcn_v2.py \
  --ckpts artifacts/models/boxing_tcn_v2fix_s2026.pt \
          artifacts/models/boxing_tcn_v2fix_s1337.pt \
          artifacts/models/boxing_tcn_v2fix_s4242.pt \
  $POST_FLAGS \
  --out submission_v2fix_3seed.csv || true

# --------- STAGE C: BIG ensemble v1(5) + v2fix(3) = 8 models ----------
echo "=== [$(date)] STAGE C: BIG 8-model ensemble ==="
python -u scripts/predict_tcn_v2.py \
  --ckpts artifacts/models/boxing_tcn_v1_s2026.pt \
          artifacts/models/boxing_tcn_v1_s1337.pt \
          artifacts/models/boxing_tcn_v1_s4242.pt \
          artifacts/models/boxing_tcn_v1_s31337.pt \
          artifacts/models/boxing_tcn_v1_s8088.pt \
          artifacts/models/boxing_tcn_v2fix_s2026.pt \
          artifacts/models/boxing_tcn_v2fix_s1337.pt \
          artifacts/models/boxing_tcn_v2fix_s4242.pt \
  $POST_FLAGS \
  --out submission_big_ensemble.csv || true

echo "=== [$(date)] DONE workplan ==="
