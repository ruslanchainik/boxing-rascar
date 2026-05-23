from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from boxing_lstm_pipeline.dataset import load_feature_npz, window_at
from boxing_lstm_pipeline.model import BoxingLSTM, CLASS_NAMES
from boxing_lstm_pipeline.paths import MODEL_DIR, POSE_DIR, SAMPLE_SUBMISSION, TEST_VIDEOS, TRAIN_PUNCHES, TRAIN_VIDEOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=Path, default=POSE_DIR)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "boxing_lstm.pt")
    parser.add_argument("--out", type=Path, default=Path("submission_lstm.csv"))
    parser.add_argument("--scan_stride", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--nms_frames", type=int, default=12)
    parser.add_argument("--cap_multiplier", type=float, default=1.10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def expected_counts(test_videos: pd.DataFrame) -> dict[str, int]:
    train_punches = pd.read_csv(TRAIN_PUNCHES)
    train_videos = pd.read_csv(TRAIN_VIDEOS)
    clear = train_punches[train_punches["clear"].astype(str).str.lower().eq("true")]
    counts = clear.groupby("video_key").size()
    train_videos = train_videos.copy()
    train_videos["round_number"] = pd.to_numeric(train_videos["round_number"], errors="coerce")
    test_videos = test_videos.copy()
    test_videos["round_number"] = pd.to_numeric(test_videos["round_number"], errors="coerce")
    train_videos["clear_count"] = train_videos["video_key"].map(counts).fillna(0).astype(int)
    train_videos["clear_rate"] = train_videos["clear_count"] / (train_videos["frame_count"] / 30.0)

    out = {}
    for _, row in test_videos.iterrows():
        pool = train_videos[
            (train_videos["dataset_type"] == row["dataset_type"])
            & (train_videos["round_number"] == row["round_number"])
            & (train_videos["clear_count"] > 0)
        ]
        if pool.empty:
            pool = train_videos[
                (train_videos["dataset_type"] == row["dataset_type"])
                & (train_videos["clear_count"] > 0)
            ]
        rate = float(pool["clear_rate"].median())
        out[row["video_key"]] = max(1, int(round(rate * (int(row["frame_count"]) / 30.0))))
    return out


@torch.no_grad()
def score_video(model: BoxingLSTM, features: np.ndarray, window: int, scan_stride: int, device: str) -> list[dict]:
    model.eval()
    candidates = []
    centers = list(range(0, len(features), scan_stride))
    batch_x = []
    batch_centers = []
    for center in centers:
        batch_x.append(window_at(features, center, window))
        batch_centers.append(center)
        if len(batch_x) == 256:
            candidates.extend(score_batch(model, batch_x, batch_centers, device))
            batch_x, batch_centers = [], []
    if batch_x:
        candidates.extend(score_batch(model, batch_x, batch_centers, device))
    return candidates


@torch.no_grad()
def score_batch(model: BoxingLSTM, batch_x: list[np.ndarray], centers: list[int], device: str) -> list[dict]:
    x = torch.from_numpy(np.stack(batch_x)).to(device)
    logits = model(x)
    probs = torch.sigmoid(logits["event"].squeeze(1)).detach().cpu().numpy()
    decoded = {}
    for name, values in CLASS_NAMES.items():
        idx = logits[name].argmax(1).detach().cpu().numpy()
        decoded[name] = [values[i] for i in idx]

    out = []
    for i, center in enumerate(centers):
        row = {"frame": int(center), "score": float(probs[i])}
        for name in CLASS_NAMES:
            row[name] = decoded[name][i]
        out.append(row)
    return out


def nms_candidates(candidates: list[dict], threshold: float, nms_frames: int, cap: int) -> list[dict]:
    candidates = [c for c in candidates if c["score"] >= threshold]
    candidates.sort(key=lambda c: c["score"], reverse=True)
    kept: list[dict] = []
    for cand in candidates:
        if any(abs(cand["frame"] - old["frame"]) <= nms_frames for old in kept):
            continue
        kept.append(cand)
        if len(kept) >= cap:
            break
    return sorted(kept, key=lambda c: c["frame"])


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = BoxingLSTM(ckpt["input_dim"], hidden_dim=ckpt["hidden_dim"]).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    window = int(ckpt["window"])

    sample = pd.read_csv(SAMPLE_SUBMISSION)
    test_videos = pd.read_csv(TEST_VIDEOS)
    exp_counts = expected_counts(test_videos)
    sample["clear"] = "false"

    output_rows = []
    for _, video in test_videos.iterrows():
        video_key = video["video_key"]
        feature_path = args.features_dir / f"{video_key}.npz"
        if not feature_path.exists():
            raise SystemExit(f"Missing features for {video_key}: {feature_path}")
        _, features = load_feature_npz(feature_path)
        candidates = score_video(model, features, window, args.scan_stride, args.device)
        cap = int(round(exp_counts[video_key] * args.cap_multiplier))
        selected = nms_candidates(candidates, args.threshold, args.nms_frames, cap)
        print(f"{video_key}: selected={len(selected)} cap={cap}")

        template_rows = sample[sample["video_key"].eq(video_key)].copy().reset_index(drop=True)
        for i, cand in enumerate(selected[: len(template_rows)]):
            row = template_rows.loc[i].copy()
            row["frame"] = int(np.clip(cand["frame"], 0, int(video["frame_count"]) - 1))
            row["fighter"] = cand["fighter"]
            row["punch_type"] = cand["punch_type"]
            row["hand"] = cand["hand"]
            row["target"] = cand["target"]
            row["effectiveness"] = cand["effectiveness"]
            row["clear"] = "true"
            output_rows.append(row)

        for i in range(len(selected), len(template_rows)):
            row = template_rows.loc[i].copy()
            row["clear"] = "false"
            output_rows.append(row)

    out = pd.DataFrame(output_rows)
    out["id"] = np.arange(1, len(out) + 1)
    out = out[pd.read_csv(SAMPLE_SUBMISSION, nrows=0).columns.tolist()]
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} rows={len(out)}")


if __name__ == "__main__":
    main()
