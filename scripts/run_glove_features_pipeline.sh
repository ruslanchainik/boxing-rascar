#!/usr/bin/env bash
set -e
cd C:/hack
export PYTHONUNBUFFERED=1

FEAT="artifacts/pose_features_glove"
POST="--no_snap --hand_prior --attr_avg_half 4 --threshold 0.65 --min_distance 6"

echo "=== [$(date)] STAGE 1: extract glove tracks (YOLO inference, all videos) ==="
python -u scripts/extract_glove_tracks.py --stride 2

echo "=== [$(date)] STAGE 2: build features with glove ==="
python -u scripts/build_features_with_glove.py

echo "=== [$(date)] STAGE 3: train 3 v1 seeds on glove features ==="
for s in 2026 1337 4242; do
  OUT="artifacts/models/boxing_tcn_glove_s${s}.pt"
  if [ -f "$OUT" ]; then echo "skip $s"; continue; fi
  python -u scripts/train_tcn.py --out "$OUT" --seed $s --epochs 30 \
    --hidden_dim 192 --batch_size 8 --crop_len 512 --crops_per_video 12 \
    --features_dir "$FEAT" || true
done

echo "=== [$(date)] STAGE 4: train 2 v2fix seeds on glove features ==="
for s in 2026 4242; do
  OUT="artifacts/models/boxing_tcn_v2fix_glove_s${s}.pt"
  if [ -f "$OUT" ]; then echo "skip v2fix $s"; continue; fi
  python -u scripts/train_tcn_v2.py --out "$OUT" --seed $s --epochs 30 \
    --dropout 0.2 --weight_decay 1e-4 --mixup_prob 0.0 --color_swap_prob 0.5 \
    --label_smoothing 0.05 --ema_decay 0.995 --crops_per_video 12 \
    --features_dir "$FEAT" || true
done

echo "=== [$(date)] DONE glove-features pipeline ==="
