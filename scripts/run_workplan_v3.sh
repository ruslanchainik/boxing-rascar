#!/usr/bin/env bash
set -e
cd C:/hack
export PYTHONUNBUFFERED=1

POST="--no_snap --hand_prior --attr_avg_half 4 --threshold 0.65 --min_distance 6"

# ============================================================
# A. RETRAIN garbage seeds with save-bug fix
# ============================================================
echo "=== [$(date)] STAGE A: retrain garbage seeds ==="

# Delete the bad ckpts so train_tcn.py recreates them
rm -f artifacts/models/boxing_tcn_v1_s1337.pt artifacts/models/boxing_tcn_v1_s8088.pt
rm -f artifacts/models/boxing_tcn_v2fix_s2026.pt artifacts/models/boxing_tcn_v2fix_s1337.pt

for s in 1337 8088; do
  OUT="artifacts/models/boxing_tcn_v1_s${s}.pt"
  echo "=== [$(date)] retrain v1 seed=$s ==="
  python -u scripts/train_tcn.py --out "$OUT" --seed $s --epochs 30 \
    --hidden_dim 192 --batch_size 8 --crop_len 512 --crops_per_video 12 || true
done

for s in 2026 1337; do
  OUT="artifacts/models/boxing_tcn_v2fix_s${s}.pt"
  echo "=== [$(date)] retrain v2fix seed=$s ==="
  python -u scripts/train_tcn_v2.py --out "$OUT" --seed $s --epochs 30 \
    --dropout 0.2 --weight_decay 1e-4 --mixup_prob 0.0 --color_swap_prob 0.5 \
    --label_smoothing 0.05 --ema_decay 0.995 --crops_per_video 12 || true
done

CKPTS="artifacts/models/boxing_tcn.pt \
  artifacts/models/boxing_tcn_v1_s2026.pt \
  artifacts/models/boxing_tcn_v1_s1337.pt \
  artifacts/models/boxing_tcn_v1_s4242.pt \
  artifacts/models/boxing_tcn_v1_s31337.pt \
  artifacts/models/boxing_tcn_v1_s8088.pt \
  artifacts/models/boxing_tcn_v2fix_s2026.pt \
  artifacts/models/boxing_tcn_v2fix_s1337.pt \
  artifacts/models/boxing_tcn_v2fix_s4242.pt"

# ============================================================
# A1. submission_strong9: ensemble of 9 ckpts, best postprocess
# ============================================================
echo "=== [$(date)] A1: ensemble strong9 ==="
python -u scripts/predict_tcn_v2.py --ckpts $CKPTS $POST --out submission_strong9.csv || true

# ============================================================
# C. submission_tta: strong9 + TTA color-swap
# ============================================================
echo "=== [$(date)] C: TTA color-swap ==="
python -u scripts/predict_tcn_v2.py --ckpts $CKPTS $POST --tta_colorswap \
  --out submission_tta_colorswap.csv || true

# ============================================================
# D. submission_classrebal: strong9 + class rebalance
# ============================================================
echo "=== [$(date)] D: class rebalance ==="
python -u scripts/predict_tcn_v2.py --ckpts $CKPTS $POST \
  --rebalance_punch_type --rebalance_effectiveness \
  --out submission_classrebal.csv || true

# ============================================================
# B. submission_perfight: strong9 + per-fight thresholds
# 3 different thresholds, slightly above/below 0.65
# ============================================================
echo "=== [$(date)] B: per-fight thresholds (uniform v1 0.6, v2 0.65, v3 0.7) ==="
python -u scripts/predict_tcn_v2.py --ckpts $CKPTS $POST \
  --per_fight_thresholds "tour1=0.6,tour2=0.65,tour3=0.7" \
  --out submission_perfight_a.csv || true

echo "=== [$(date)] B': per-fight thresholds (uniform v1 0.7, v2 0.65, v3 0.6) ==="
python -u scripts/predict_tcn_v2.py --ckpts $CKPTS $POST \
  --per_fight_thresholds "tour1=0.7,tour2=0.65,tour3=0.6" \
  --out submission_perfight_b.csv || true

# ============================================================
# Combined ALL: TTA + rebalance
# ============================================================
echo "=== [$(date)] FINAL: strong9 + TTA + rebalance ==="
python -u scripts/predict_tcn_v2.py --ckpts $CKPTS $POST --tta_colorswap \
  --rebalance_punch_type --rebalance_effectiveness \
  --out submission_final_combo.csv || true

echo "=== [$(date)] DONE workplan_v3 ==="
