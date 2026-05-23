"""Auto-label glove bboxes from wrist keypoints, save as YOLO dataset.

For each video, sample N frames. For each frame, take wrist keypoints
(red L/R, blue L/R) from saved pose features. Where conf > thr, place
a bbox of size ~5% of frame width around the wrist. Save image and YOLO label.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from boxing_lstm_pipeline.paths import POSE_DIR, TRAIN_VIDEOS, TEST_VIDEOS


# Wrist keypoint coord offsets within base features
RED_LW_X, RED_LW_Y, RED_LW_C = 27, 28, 29
RED_RW_X, RED_RW_Y, RED_RW_C = 30, 31, 32
BLUE_LW_X, BLUE_LW_Y, BLUE_LW_C = 59 + 27, 59 + 28, 59 + 29
BLUE_RW_X, BLUE_RW_Y, BLUE_RW_C = 59 + 30, 59 + 31, 59 + 32


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=Path("C:/hack/artifacts/glove_dataset"))
    p.add_argument("--frames_per_video", type=int, default=24)
    p.add_argument("--bbox_size_pct", type=float, default=0.05, help="bbox side as fraction of frame width")
    p.add_argument("--min_conf", type=float, default=0.5)
    p.add_argument("--val_frac", type=float, default=0.1)
    return p.parse_args()


def video_path_for(row):
    return Path("C:/hack") / row["video_path"]


def main():
    args = parse_args()
    img_train = args.out_dir / "images" / "train"
    img_val = args.out_dir / "images" / "val"
    lbl_train = args.out_dir / "labels" / "train"
    lbl_val = args.out_dir / "labels" / "val"
    for d in (img_train, img_val, lbl_train, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2026)
    manifest = pd.concat([pd.read_csv(TRAIN_VIDEOS), pd.read_csv(TEST_VIDEOS)], ignore_index=True)

    saved_imgs = 0; saved_boxes = 0
    for _, row in manifest.iterrows():
        vk = row["video_key"]
        vp = video_path_for(row)
        if not vp.exists():
            continue
        npz_path = POSE_DIR / f"{vk}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        base = data["base_features"].astype(np.float32)
        T = len(base)
        if T < 50: continue

        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened(): continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames uniformly (skip first/last 10%)
        step = max(1, (T - T // 5) // args.frames_per_video)
        sample_idx = list(range(T // 10, T - T // 10, step))[:args.frames_per_video]

        bbox_side = int(args.bbox_size_pct * width)

        for fi in sample_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok: continue
            fb = base[fi]
            boxes = []
            for x_idx, y_idx, c_idx in [
                (RED_LW_X, RED_LW_Y, RED_LW_C),
                (RED_RW_X, RED_RW_Y, RED_RW_C),
                (BLUE_LW_X, BLUE_LW_Y, BLUE_LW_C),
                (BLUE_RW_X, BLUE_RW_Y, BLUE_RW_C),
            ]:
                cx = float(fb[x_idx]) * width
                cy = float(fb[y_idx]) * height
                conf = float(fb[c_idx])
                if conf < args.min_conf or cx <= 0 or cy <= 0:
                    continue
                # bbox center; clip to image
                x1 = max(0, int(cx - bbox_side / 2))
                y1 = max(0, int(cy - bbox_side / 2))
                x2 = min(width, int(cx + bbox_side / 2))
                y2 = min(height, int(cy + bbox_side / 2))
                if x2 - x1 < 8 or y2 - y1 < 8: continue
                # YOLO format: cx cy w h normalized
                ncx = (x1 + x2) / 2 / width
                ncy = (y1 + y2) / 2 / height
                nw = (x2 - x1) / width
                nh = (y2 - y1) / height
                boxes.append((0, ncx, ncy, nw, nh))  # single class "glove"
            if not boxes: continue

            split = "val" if rng.random() < args.val_frac else "train"
            img_dir = img_val if split == "val" else img_train
            lbl_dir = lbl_val if split == "val" else lbl_train
            stem = f"{vk}_f{fi:06d}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with open(lbl_dir / f"{stem}.txt", "w") as f:
                for b in boxes:
                    f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
            saved_imgs += 1
            saved_boxes += len(boxes)
        cap.release()
        print(f"  {vk}: sampled {len(sample_idx)} frames")

    # Write dataset yaml
    yaml_path = args.out_dir / "glove.yaml"
    yaml_path.write_text(
        f"path: {args.out_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: glove\n",
        encoding="utf-8",
    )
    print(f"\n[done] images={saved_imgs}, boxes={saved_boxes}, yaml={yaml_path}")


if __name__ == "__main__":
    main()
