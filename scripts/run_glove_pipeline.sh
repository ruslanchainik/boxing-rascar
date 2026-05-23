#!/usr/bin/env bash
set -e
cd C:/hack
export PYTHONUNBUFFERED=1

DATA="C:/hack/artifacts/glove_dataset"
RUNS="C:/hack/artifacts/glove_runs"

echo "=== [$(date)] STAGE 1: build glove dataset (auto-label) ==="
if [ ! -d "$DATA/images/train" ] || [ -z "$(ls -A $DATA/images/train 2>/dev/null)" ]; then
  python -u scripts/build_glove_dataset.py --out_dir "$DATA" --frames_per_video 24
else
  echo "skip: dataset exists"
fi

echo "=== [$(date)] STAGE 2: train YOLOv8n on glove dataset ==="
python -u -c "
from ultralytics import YOLO
import os
os.makedirs('$RUNS', exist_ok=True)
m = YOLO('yolov8n.pt')
m.train(
    data='$DATA/glove.yaml',
    epochs=50,
    imgsz=480,
    batch=8,
    workers=0,
    device=0,
    project='$RUNS',
    name='glove_yolov8n',
    patience=10,
    save=True,
    plots=False,
    cache=False,
    verbose=True,
    amp=False,
)
"

echo "=== [$(date)] DONE glove pipeline ==="
echo "Best weights: $RUNS/glove_yolov8n/weights/best.pt"
