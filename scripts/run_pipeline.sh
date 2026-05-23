#!/usr/bin/env bash
set -e
cd C:/hack
export PYTHONUNBUFFERED=1
export YOLO_OFFLINE=true
echo "=== [$(date)] STEP 1/4: extract train pose ==="
python -u scripts/extract_pose_features.py --split train --model yolo11n-pose.pt --imgsz 640 --stride 4 --device 0 --out_dir artifacts/pose_features
echo "=== [$(date)] STEP 2/4: extract test pose ==="
python -u scripts/extract_pose_features.py --split test --model yolo11n-pose.pt --imgsz 640 --stride 4 --device 0 --out_dir artifacts/pose_features
echo "=== [$(date)] STEP 3/4: train TCN ==="
python -u scripts/train_tcn.py --epochs 30 --batch_size 8 --crop_len 512 --crops_per_video 12 --device cuda
echo "=== [$(date)] STEP 4/4: predict ==="
python -u scripts/predict_tcn.py --threshold 0.35 --min_distance 10 --out submission_tcn.csv
echo "=== [$(date)] DONE ==="
