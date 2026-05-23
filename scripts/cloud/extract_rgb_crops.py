"""Extract ring-region 224x398 crops from all videos and save as small mp4 at fps=12.5.

This runs LOCALLY (uses pose features to determine ring bbox).
Output crops are uploaded to DataSphere for RGB model training.

Output: artifacts/rgb_crops/<video_key>.mp4  (224x398 @ 12.5 fps, ~5 MB each)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from boxing_lstm_pipeline.paths import POSE_DIR, TRAIN_VIDEOS, TEST_VIDEOS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=Path("C:/hack/artifacts/rgb_crops"))
    p.add_argument("--crop_w", type=int, default=398)
    p.add_argument("--crop_h", type=int, default=224)
    p.add_argument("--target_fps", type=float, default=12.5)
    p.add_argument("--margin", type=float, default=0.15, help="extra margin around ring bbox")
    return p.parse_args()


def video_path_for(row):
    return Path("C:/hack") / row["video_path"]


def estimate_ring_bbox(base: np.ndarray, width: int, height: int, margin: float) -> tuple[int,int,int,int]:
    """Use pose features to find rough ring bbox = encompasses both fighters across video."""
    # red bbox: cols 51..54 (cx, cy, w, h normalized)
    # blue bbox: cols 110..113
    T = len(base)
    valid = base[:, 58] + base[:, 117]  # presence sums
    use = valid > 0
    if not use.any():
        return 0, 0, width, height
    red_cx = base[use, 51] * width; red_cy = base[use, 52] * height
    blue_cx = base[use, 110] * width; blue_cy = base[use, 111] * height
    all_x = np.concatenate([red_cx, blue_cx])
    all_y = np.concatenate([red_cy, blue_cy])
    cx_min, cx_max = float(all_x.min()), float(all_x.max())
    cy_min, cy_max = float(all_y.min()), float(all_y.max())
    bw = cx_max - cx_min; bh = cy_max - cy_min
    cx_min -= bw * margin; cx_max += bw * margin
    cy_min -= bh * margin; cy_max += bh * margin
    x1 = max(0, int(cx_min)); y1 = max(0, int(cy_min))
    x2 = min(width, int(cx_max)); y2 = min(height, int(cy_max))
    if x2 - x1 < 100 or y2 - y1 < 100:
        return 0, 0, width, height
    return x1, y1, x2, y2


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.concat([pd.read_csv(TRAIN_VIDEOS), pd.read_csv(TEST_VIDEOS)], ignore_index=True)

    for _, row in manifest.iterrows():
        vk = row["video_key"]
        out_path = args.out_dir / f"{vk}.mp4"
        if out_path.exists():
            print(f"skip {vk}"); continue
        vp = video_path_for(row)
        if not vp.exists():
            print(f"missing {vp}"); continue
        npz_pose = POSE_DIR / f"{vk}.npz"
        if not npz_pose.exists():
            print(f"missing pose {vk}"); continue
        base = np.load(npz_pose)["base_features"].astype(np.float32)

        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"cant open {vp}"); continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        x1, y1, x2, y2 = estimate_ring_bbox(base, width, height, args.margin)
        print(f"{vk}: source {width}x{height} fps={fps:.1f} crop=({x1},{y1})-({x2},{y2})")

        # Resample to target fps
        skip = max(1, int(round(fps / args.target_fps)))
        eff_fps = fps / skip
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, eff_fps, (args.crop_w, args.crop_h))
        if not writer.isOpened():
            print(f"cant write {out_path}"); cap.release(); continue

        idx = 0
        kept = 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            if idx % skip == 0:
                crop = fr[y1:y2, x1:x2]
                if crop.size == 0:
                    crop = fr
                resized = cv2.resize(crop, (args.crop_w, args.crop_h), interpolation=cv2.INTER_AREA)
                writer.write(resized)
                kept += 1
            idx += 1
        cap.release(); writer.release()
        print(f"  saved {out_path.name}: {kept} frames @ {eff_fps:.1f} fps")


if __name__ == "__main__":
    main()
