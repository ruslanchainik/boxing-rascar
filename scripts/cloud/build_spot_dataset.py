"""Build SPOT-compatible boxing dataset from our punches.csv + rgb_crops mp4 metadata.

Outputs to cloud_spot/data/boxing/:
  - train.json  (~50 train videos, with clear=true events)
  - val.json    (~10 val videos)
  - test.json   (9 test videos, empty events)

Class encoding: fighter_type (8 classes: red_jab .. blue_uppercut).
Frame indices in 15 fps space (rgb_crops are at 15 fps).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from boxing_lstm_pipeline.paths import POSE_DIR, TRAIN_PUNCHES, TRAIN_VIDEOS, TEST_VIDEOS

CROPS_DIR = Path("C:/hack/artifacts/rgb_crops")
OUT_DIR = Path("C:/hack/cloud_spot/data/boxing")
SPOT_FPS = 15.0   # rgb_crops were resampled to 15 fps
VAL_RATIO = 0.15
SEED = 2026


def video_info(vk: str) -> tuple[int, float, int, int] | None:
    """Return (num_spot_frames, fps_native, width, height) for video_key."""
    mp4 = CROPS_DIR / f"{vk}.mp4"
    pose = POSE_DIR / f"{vk}.npz"
    if not mp4.exists() or not pose.exists():
        return None
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return None
    n_spot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n_spot <= 0:
        return None
    p = np.load(pose)
    fps_native = float(p["fps"]) if "fps" in p.files else 30.0
    width = int(p["width"]) if "width" in p.files else 1920
    height = int(p["height"]) if "height" in p.files else 1080
    return n_spot, fps_native, width, height


def build_record(vk: str, info, punches_for_vk: pd.DataFrame) -> dict:
    n_spot, fps_native, width, height = info
    events = []
    if len(punches_for_vk) > 0:
        clear = punches_for_vk["clear"].astype(str).str.lower() == "true"
        rows_iter = punches_for_vk[clear].iterrows()
    else:
        rows_iter = iter([])
    for _, row in rows_iter:
        src_frame = int(row["frame"])
        spot_frame = int(round(src_frame * SPOT_FPS / fps_native))
        if not (0 <= spot_frame < n_spot):
            continue
        label = f'{row["fighter"]}_{row["punch_type"]}'
        events.append({"frame": spot_frame, "label": label, "comment": ""})
    return {
        "video": vk,
        "num_frames": n_spot,
        "num_events": len(events),
        "events": events,
        "fps": SPOT_FPS,
        "width": width,
        "height": height,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    punches = pd.read_csv(TRAIN_PUNCHES)
    train_manifest = pd.read_csv(TRAIN_VIDEOS)
    test_manifest = pd.read_csv(TEST_VIDEOS)

    # Train + val split by fight (group rounds together to avoid leak)
    rng = np.random.default_rng(SEED)
    fight_keys = sorted(set(zip(train_manifest["data_root"], train_manifest["fight_folder"].fillna(""))))
    rng.shuffle(fight_keys)
    n_val_fights = max(1, int(len(fight_keys) * VAL_RATIO))
    val_fights = set(fight_keys[:n_val_fights])

    train_records = []
    val_records = []
    skipped = 0
    for _, vrow in train_manifest.iterrows():
        vk = vrow["video_key"]
        info = video_info(vk)
        if info is None:
            print(f"skip {vk} (no crop/pose)")
            skipped += 1
            continue
        key = (vrow["data_root"], str(vrow["fight_folder"]) if not pd.isna(vrow["fight_folder"]) else "")
        rec = build_record(vk, info, punches[punches["video_key"] == vk])
        if key in val_fights:
            val_records.append(rec)
        else:
            train_records.append(rec)

    test_records = []
    for _, vrow in test_manifest.iterrows():
        vk = vrow["video_key"]
        info = video_info(vk)
        if info is None:
            print(f"skip test {vk}")
            continue
        rec = build_record(vk, info, pd.DataFrame())  # no labels
        test_records.append(rec)

    (OUT_DIR / "train.json").write_text(json.dumps(train_records, indent=2), encoding="utf-8")
    (OUT_DIR / "val.json").write_text(json.dumps(val_records, indent=2), encoding="utf-8")
    (OUT_DIR / "test.json").write_text(json.dumps(test_records, indent=2), encoding="utf-8")

    print(f"[done] train={len(train_records)} (events={sum(r['num_events'] for r in train_records)})")
    print(f"       val={len(val_records)} (events={sum(r['num_events'] for r in val_records)})")
    print(f"       test={len(test_records)}")
    print(f"       skipped={skipped}")


if __name__ == "__main__":
    main()
