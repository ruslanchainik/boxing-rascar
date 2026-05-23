from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parents[1]))

from boxing_lstm_pipeline.paths import POSE_DIR, TEST_VIDEOS, TRAIN_VIDEOS
from boxing_lstm_pipeline.pose_features import (
    BASE_DIM,
    RoleAssigner,
    add_temporal_deltas,
    frame_feature,
    video_path_from_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--video_key", nargs="*", default=None)
    parser.add_argument("--model", default="yolo11m-pose.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--out_dir", type=Path, default=POSE_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def manifest_for_split(split: str) -> pd.DataFrame:
    path = TRAIN_VIDEOS if split == "train" else TEST_VIDEOS
    return pd.read_csv(path)


def extract_video(row: pd.Series, model: YOLO, args: argparse.Namespace) -> bool:
    out_path = args.out_dir / f"{row.video_key}.npz"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        print(f"[extract] skip {row.video_key}: already exists")
        return True
    video_path = video_path_from_manifest(row)
    if not video_path.exists():
        print(f"[extract] SKIP missing file: {video_path}")
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[extract] SKIP cannot open: {video_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(row.width)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(row.height)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or row.frame_count)

    assigner = RoleAssigner()
    base_features: list[np.ndarray] = []
    last_feature = np.zeros(BASE_DIM, dtype=np.float32)

    frame_idx = 0
    print(f"[extract] {row.video_key}: {video_path} ({frame_count} frames)")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

        if frame_idx % max(1, args.stride) == 0:
            results = model.track(
                frame,
                persist=True,
                classes=[0],
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
                tracker=str(Path(__file__).resolve().parent / "bytetrack_static.yaml"),
            )
            assigned = {}
            if results and results[0].boxes is not None and results[0].keypoints is not None:
                boxes_obj = results[0].boxes
                bboxes = boxes_obj.xyxy.cpu().numpy()
                if boxes_obj.id is None:
                    track_ids = np.arange(len(bboxes))
                else:
                    track_ids = boxes_obj.id.int().cpu().numpy()
                keypoints = results[0].keypoints.data.cpu().numpy()
                assigned = assigner.update(frame, track_ids, bboxes, keypoints)
            last_feature = frame_feature(assigned, width, height)

        base_features.append(last_feature.copy())
        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"  {row.video_key}: {frame_idx}/{frame_count}")

    cap.release()
    base = np.stack(base_features).astype(np.float32) if base_features else np.zeros((0, BASE_DIM), dtype=np.float32)
    features = add_temporal_deltas(base)
    np.savez_compressed(
        out_path,
        frames=np.arange(len(features), dtype=np.int32),
        features=features,
        base_features=base,
        width=np.array(width),
        height=np.array(height),
        fps=np.array(fps),
    )
    print(f"[extract] saved {out_path} shape={features.shape}")
    return True


def main() -> None:
    args = parse_args()
    manifest = manifest_for_split(args.split)
    if args.video_key:
        manifest = manifest[manifest["video_key"].isin(args.video_key)].copy()
    if manifest.empty:
        raise SystemExit("No videos selected")

    model = YOLO(args.model)
    for _, row in manifest.iterrows():
        extract_video(row, model, args)


if __name__ == "__main__":
    main()
