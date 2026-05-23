from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from boxing_lstm_pipeline.tcn_dataset import ATTR_NAMES
from boxing_lstm_pipeline.tcn_model import BoxingTCN, CLASS_NAMES
from boxing_lstm_pipeline.paths import (
    MODEL_DIR, POSE_DIR, SAMPLE_SUBMISSION, TEST_VIDEOS,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", type=Path, default=POSE_DIR)
    p.add_argument("--ckpt", type=Path, default=MODEL_DIR / "boxing_tcn.pt")
    p.add_argument("--out", type=Path, default=Path("submission_tcn.csv"))
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--min_distance", type=int, default=10)
    p.add_argument("--max_predictions_per_video", type=int, default=200)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def peak_pick(scores: np.ndarray, threshold: float, min_distance: int) -> list[tuple[int, float]]:
    """Local-maxima peak picking with NMS."""
    T = len(scores)
    candidates = []
    for i in range(T):
        if scores[i] < threshold:
            continue
        lo = max(0, i - min_distance)
        hi = min(T, i + min_distance + 1)
        if scores[i] >= scores[lo:hi].max() - 1e-6:
            candidates.append((i, float(scores[i])))
    # NMS by descending score
    candidates.sort(key=lambda x: -x[1])
    kept = []
    used = np.zeros(T, dtype=bool)
    for f, s in candidates:
        lo = max(0, f - min_distance)
        hi = min(T, f + min_distance + 1)
        if used[lo:hi].any():
            continue
        kept.append((f, s))
        used[lo:hi] = True
    kept.sort(key=lambda x: x[0])
    return kept


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = BoxingTCN(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        n_blocks=ckpt["n_blocks"],
        dropout=ckpt.get("dropout", 0.15),
    ).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[predict] loaded {args.ckpt} score={ckpt.get('score', 'n/a')}")

    test_videos = pd.read_csv(TEST_VIDEOS)
    rows = []
    for _, vrow in test_videos.iterrows():
        vk = vrow["video_key"]
        fpath = args.features_dir / f"{vk}.npz"
        if not fpath.exists():
            print(f"[predict] missing features for {vk}")
            continue
        data = np.load(fpath)
        feats = data["features"].astype(np.float32)
        fps_native = float(data["fps"]) if "fps" in data.files else 30.0
        T = len(feats)

        with torch.no_grad():
            # chunked inference to fit memory
            chunk = 4096
            event_scores = np.zeros(T, dtype=np.float32)
            attr_logits = {n: np.zeros((T, len(CLASS_NAMES[n])), dtype=np.float32) for n in ATTR_NAMES}
            for s in range(0, T, chunk):
                e = min(T, s + chunk)
                x = torch.from_numpy(feats[s:e]).unsqueeze(0).to(args.device)
                out = model(x)
                event_scores[s:e] = torch.sigmoid(out["event"]).squeeze(0).cpu().numpy()
                for n in ATTR_NAMES:
                    attr_logits[n][s:e] = out[n].squeeze(0).cpu().numpy()

        peaks = peak_pick(event_scores, args.threshold, args.min_distance)
        peaks = peaks[: args.max_predictions_per_video]
        print(f"[predict] {vk}: {len(peaks)} peaks (T={T}, fps={fps_native:.2f})")

        for native_frame, score in peaks:
            # convert native frame to 30fps frame as the metric uses
            t_sec = native_frame / max(fps_native, 1e-3)
            metric_frame = int(round(t_sec * 30.0))
            row = {
                "video_id": vrow["video_id"],
                "agn_index": int(vrow["agn_index"]),
                "video_key": vk,
                "frame": metric_frame,
                "score": score,
            }
            for n in ATTR_NAMES:
                cls_idx = int(attr_logits[n][native_frame].argmax())
                row[n] = CLASS_NAMES[n][cls_idx]
            rows.append(row)

    pred_df = pd.DataFrame(rows)
    # remap effectiveness back to submission schema {landed,blocked,miss} — already those names
    pred_df["clear"] = "true"

    # pad/truncate to sample submission size
    sample = pd.read_csv(SAMPLE_SUBMISSION)
    n_required = len(sample)
    if len(pred_df) > n_required:
        pred_df = pred_df.sort_values("score", ascending=False).head(n_required)
    pad_rows = n_required - len(pred_df)
    if pad_rows > 0:
        pad_df = sample.iloc[len(pred_df):].copy()
        pad_df["clear"] = "false"
        pred_df = pd.concat([pred_df.drop(columns=["score"]), pad_df.drop(columns=["id"])], ignore_index=True)
    else:
        pred_df = pred_df.drop(columns=["score"])

    pred_df = pred_df.reset_index(drop=True)
    pred_df.insert(0, "id", np.arange(1, len(pred_df) + 1))
    cols = ["id", "video_id", "agn_index", "video_key", "frame", "fighter",
            "punch_type", "hand", "target", "effectiveness", "clear"]
    pred_df = pred_df[cols]
    pred_df.to_csv(args.out, index=False)
    n_real = int((pred_df["clear"].astype(str).str.lower() == "true").sum())
    print(f"[predict] wrote {args.out} rows={len(pred_df)} clear_true={n_real}")


if __name__ == "__main__":
    main()
