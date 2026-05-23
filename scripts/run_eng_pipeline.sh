#!/usr/bin/env bash
set -e
cd C:/hack
export PYTHONUNBUFFERED=1

FEAT="artifacts/pose_features_eng"
POST="--no_snap --hand_prior --attr_avg_half 4 --threshold 0.65 --min_distance 6"

# --------- Train 3 v1 seeds on engineered features ---------
echo "=== [$(date)] STAGE 1: train v1 ENG features (3 seeds) ==="
for s in 2026 1337 4242; do
  OUT="artifacts/models/boxing_tcn_eng_s${s}.pt"
  if [ -f "$OUT" ]; then
    echo "skip $s"; continue
  fi
  echo "=== [$(date)] train eng seed=$s ==="
  python -u scripts/train_tcn.py --out "$OUT" --seed $s --epochs 30 \
    --hidden_dim 192 --batch_size 8 --crop_len 512 --crops_per_video 12 \
    --features_dir "$FEAT" || true
done

# --------- Train 2 v2fix seeds on engineered features ---------
echo "=== [$(date)] STAGE 2: train v2fix ENG (2 seeds) ==="
for s in 2026 4242; do
  OUT="artifacts/models/boxing_tcn_v2fix_eng_s${s}.pt"
  if [ -f "$OUT" ]; then
    echo "skip v2fix $s"; continue
  fi
  echo "=== [$(date)] train v2fix eng seed=$s ==="
  python -u scripts/train_tcn_v2.py --out "$OUT" --seed $s --epochs 30 \
    --dropout 0.2 --weight_decay 1e-4 --mixup_prob 0.0 --color_swap_prob 0.5 \
    --label_smoothing 0.05 --ema_decay 0.995 --crops_per_video 12 \
    --features_dir "$FEAT" || true
done

# --------- Predict: ENG-only ensemble + TTA ---------
echo "=== [$(date)] PREDICT ENG-only ensemble + TTA ==="
ENG_CKPTS="artifacts/models/boxing_tcn_eng_s2026.pt \
  artifacts/models/boxing_tcn_eng_s1337.pt \
  artifacts/models/boxing_tcn_eng_s4242.pt \
  artifacts/models/boxing_tcn_v2fix_eng_s2026.pt \
  artifacts/models/boxing_tcn_v2fix_eng_s4242.pt"

python -u scripts/predict_tcn_v2.py --ckpts $ENG_CKPTS $POST --tta_colorswap \
  --features_dir "$FEAT" \
  --out submission_eng_tta.csv || true

# --------- Predict: ENG-only ensemble (no TTA) ---------
python -u scripts/predict_tcn_v2.py --ckpts $ENG_CKPTS $POST \
  --features_dir "$FEAT" \
  --out submission_eng_only.csv || true

echo "=== [$(date)] DONE eng pipeline ==="
