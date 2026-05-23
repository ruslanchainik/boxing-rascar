"""Predict on test with TCN-RGB and output submission."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[0]))
from train_tcn_rgb import BoxingTCN, ATTR_NAMES, CLASS_NAMES


def peak_pick(scores, threshold, min_distance):
    T = len(scores); cands = []
    for i in range(T):
        if scores[i] < threshold: continue
        lo = max(0, i-min_distance); hi = min(T, i+min_distance+1)
        if scores[i] >= scores[lo:hi].max() - 1e-6:
            cands.append((i, float(scores[i])))
    cands.sort(key=lambda x: -x[1])
    used = np.zeros(T, dtype=bool); kept = []
    for f, s in cands:
        lo = max(0, f-min_distance); hi = min(T, f+min_distance+1)
        if used[lo:hi].any(): continue
        kept.append((f, s)); used[lo:hi] = True
    kept.sort(key=lambda x: x[0]); return kept


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("output/best.pt"))
    p.add_argument("--pose_dir", type=Path, default=Path("input/pose_features_glove"))
    p.add_argument("--cnn_dir", type=Path, default=Path("input/cnn_features"))
    p.add_argument("--test_videos", type=Path, default=Path("input/test/videos.csv"))
    p.add_argument("--sample", type=Path, default=Path("input/sample_submission.csv"))
    p.add_argument("--out", type=Path, default=Path("output/submission.csv"))
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--min_distance", type=int, default=6)
    p.add_argument("--attr_avg_half", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = BoxingTCN(ck["input_dim"], ck["hidden_dim"], ck["n_blocks"], ck.get("dropout", 0.2)).to(args.device)
    model.load_state_dict(ck["state_dict"]); model.eval()

    test_videos = pd.read_csv(args.test_videos)
    rows = []
    for _, vrow in test_videos.iterrows():
        vk = vrow["video_key"]
        pp = args.pose_dir / f"{vk}.npz"; cp = args.cnn_dir / f"{vk}.npz"
        if not (pp.exists() and cp.exists()):
            print(f"missing {vk}"); continue
        pose = np.load(pp)["features"].astype(np.float32)
        cnn = np.load(cp)["features"].astype(np.float32)
        fps_native = float(np.load(pp)["fps"]) if "fps" in np.load(pp).files else 30.0
        T = min(len(pose), len(cnn))
        feats = np.concatenate([pose[:T], cnn[:T]], axis=1)

        ev = np.zeros(T, np.float32)
        attrs = {n: np.zeros((T, len(CLASS_NAMES[n])), np.float32) for n in ATTR_NAMES}
        chunk = 4096
        with torch.no_grad():
            for s in range(0, T, chunk):
                e = min(T, s+chunk)
                x = torch.from_numpy(feats[s:e]).unsqueeze(0).to(args.device)
                out = model(x)
                ev[s:e] = torch.sigmoid(out["event"]).squeeze(0).cpu().numpy()
                for n in ATTR_NAMES:
                    attrs[n][s:e] = out[n].squeeze(0).cpu().numpy()

        peaks = peak_pick(ev, args.threshold, args.min_distance)[:200]
        print(f"[predict] {vk}: {len(peaks)} peaks (T={T})")
        h = args.attr_avg_half
        for fr, score in peaks:
            lo = max(0, fr - h); hi = min(T, fr + h + 1)
            attr_decided = {n: int(attrs[n][lo:hi].mean(0).argmax()) for n in ATTR_NAMES}
            metric_frame = int(round(fr / max(fps_native, 1e-3) * 30.0))
            rows.append({
                "video_id": vrow["video_id"], "agn_index": int(vrow["agn_index"]),
                "video_key": vk, "frame": metric_frame, "score": score,
                **{n: CLASS_NAMES[n][attr_decided[n]] for n in ATTR_NAMES},
                "clear": "true",
            })

    pred_df = pd.DataFrame(rows)
    sample = pd.read_csv(args.sample)
    n_req = len(sample)
    if len(pred_df) > n_req:
        pred_df = pred_df.sort_values("score", ascending=False).head(n_req)
    keep = ["video_id","agn_index","video_key","frame","fighter","punch_type","hand","target","effectiveness","clear"]
    pred_df = pred_df[keep].reset_index(drop=True)
    if len(pred_df) < n_req:
        pad = sample.iloc[len(pred_df):].copy().drop(columns=["id"])
        pad["clear"] = "false"
        pred_df = pd.concat([pred_df, pad], ignore_index=True)
    pred_df.insert(0, "id", np.arange(1, len(pred_df) + 1))
    pred_df.to_csv(args.out, index=False)
    n_real = int((pred_df["clear"].astype(str).str.lower() == "true").sum())
    print(f"wrote {args.out} rows={len(pred_df)} clear_true={n_real}")


if __name__ == "__main__":
    main()
