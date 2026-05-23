"""Run glove YOLO on all videos, associate detections with red/blue wrists, save per-frame features."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parents[1]))
from boxing_lstm_pipeline.paths import POSE_DIR, TRAIN_VIDEOS, TEST_VIDEOS

# wrist kpt coord offsets in base features
WRIST_OFFSETS = {  # (x_idx, y_idx, conf_idx)
    ("red", "left"):  (27, 28, 29),
    ("red", "right"): (30, 31, 32),
    ("blue", "left"): (59 + 27, 59 + 28, 59 + 29),
    ("blue", "right"):(59 + 30, 59 + 31, 59 + 32),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=Path("C:/hack/artifacts/glove_runs/glove_yolov8n-2/weights/best.pt"))
    p.add_argument("--out_dir", type=Path, default=Path("C:/hack/artifacts/glove_tracks"))
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=480)
    p.add_argument("--max_assoc_px", type=float, default=0.08, help="max assoc distance as frac of frame diag")
    return p.parse_args()


def video_path_for(row):
    return Path("C:/hack") / row["video_path"]


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))

    manifest = pd.concat([pd.read_csv(TRAIN_VIDEOS), pd.read_csv(TEST_VIDEOS)], ignore_index=True)

    for _, row in manifest.iterrows():
        vk = row["video_key"]
        out_path = args.out_dir / f"{vk}.npz"
        if out_path.exists():
            print(f"skip {vk}")
            continue
        vp = video_path_for(row)
        if not vp.exists():
            print(f"missing video {vp}")
            continue
        npz_pose = POSE_DIR / f"{vk}.npz"
        if not npz_pose.exists():
            print(f"missing pose {vk}")
            continue
        pose_data = np.load(npz_pose)
        base = pose_data["base_features"].astype(np.float32)
        T = len(base)

        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"cannot open {vp}"); continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        diag = float(np.hypot(width, height))

        # Output: per frame per (fighter,hand) — refined cx,cy, conf, bbox_size, has_glove
        # 4 wrists × 5 features = 20 per frame
        out = np.zeros((T, 20), dtype=np.float32)

        cur_glove_positions = {k: None for k in WRIST_OFFSETS}  # last known per wrist
        last_features = np.zeros(20, dtype=np.float32)

        frame_idx = 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            if frame_idx >= T: break
            if frame_idx % args.stride == 0:
                # Run YOLO
                results = model.predict(fr, conf=args.conf, imgsz=args.imgsz, device=0, verbose=False)
                detections = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    for bb, cc in zip(boxes, confs):
                        x1, y1, x2, y2 = bb
                        cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
                        sz = max(x2 - x1, y2 - y1)
                        detections.append((cx, cy, float(cc), float(sz)))

                # Get wrist targets from pose base (normalized; un-normalize)
                wrist_targets = {}
                for key, (xi, yi, ci) in WRIST_OFFSETS.items():
                    wx = float(base[frame_idx, xi]) * width
                    wy = float(base[frame_idx, yi]) * height
                    wconf = float(base[frame_idx, ci])
                    if wconf > 0.2 and wx > 0 and wy > 0:
                        wrist_targets[key] = (wx, wy)

                # Associate each detection to nearest wrist (greedy)
                feat_dict = {}  # key -> (cx, cy, conf, size)
                assigned_dets = set()
                for key, (wx, wy) in wrist_targets.items():
                    best_d = args.max_assoc_px * diag
                    best_i = -1
                    for i, (dx, dy, dc, dsz) in enumerate(detections):
                        if i in assigned_dets: continue
                        d = float(np.hypot(dx - wx, dy - wy))
                        if d < best_d:
                            best_d = d; best_i = i
                    if best_i >= 0:
                        dx, dy, dc, dsz = detections[best_i]
                        feat_dict[key] = (dx / width, dy / height, dc, dsz / diag)
                        assigned_dets.add(best_i)
                        cur_glove_positions[key] = feat_dict[key]
                    else:
                        # Fall back to last known
                        feat_dict[key] = cur_glove_positions[key] or (0.0, 0.0, 0.0, 0.0)

                # Build features in fixed order
                feats = []
                for key in [("red", "left"), ("red", "right"), ("blue", "left"), ("blue", "right")]:
                    cx, cy, cc, sz = feat_dict.get(key, (0, 0, 0, 0))
                    has = 1.0 if cc > 0 else 0.0
                    feats.extend([cx, cy, cc, sz, has])
                last_features = np.array(feats, dtype=np.float32)

            out[frame_idx] = last_features
            frame_idx += 1
            if frame_idx % 500 == 0:
                print(f"  {vk}: {frame_idx}/{T}")
        cap.release()
        np.savez_compressed(out_path, glove=out)
        print(f"saved {out_path.name}: shape={out.shape}")


if __name__ == "__main__":
    main()
