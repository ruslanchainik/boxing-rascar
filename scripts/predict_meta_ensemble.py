"""Meta-ensemble: combine predictions from models with different feature dims."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from boxing_lstm_pipeline.tcn_dataset import ATTR_NAMES
from boxing_lstm_pipeline.tcn_dataset_v2 import color_swap_features
from boxing_lstm_pipeline.tcn_model import BoxingTCN, CLASS_NAMES
from boxing_lstm_pipeline.paths import SAMPLE_SUBMISSION, TEST_VIDEOS


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


def load_models(ckpt_paths, device):
    models = []
    for c in ckpt_paths:
        ck = torch.load(c, map_location=device, weights_only=False)
        m = BoxingTCN(input_dim=ck["input_dim"], hidden_dim=ck["hidden_dim"],
                      n_blocks=ck["n_blocks"], dropout=ck.get("dropout", 0.15)).to(device)
        m.load_state_dict(ck["state_dict"]); m.eval()
        models.append((m, ck["input_dim"]))
    return models


def forward_group(models, feats, device, tta=False, chunk=4096):
    T = len(feats)
    ev = np.zeros(T, np.float32)
    attrs = {n: np.zeros((T, len(CLASS_NAMES[n])), np.float32) for n in ATTR_NAMES}
    variants = [(feats, False)]
    if tta:
        variants.append((color_swap_features(feats), True))
    n_runs = 0
    with torch.no_grad():
        for var_feats, is_swapped in variants:
            for model, in_dim in models:
                if var_feats.shape[1] != in_dim:
                    continue
                for s in range(0, T, chunk):
                    e = min(T, s+chunk)
                    x = torch.from_numpy(var_feats[s:e]).unsqueeze(0).to(device)
                    out = model(x)
                    ev[s:e] += torch.sigmoid(out["event"]).squeeze(0).cpu().numpy()
                    for n in ATTR_NAMES:
                        l = out[n].squeeze(0).cpu().numpy()
                        if is_swapped and n == "fighter":
                            l = l[:, ::-1].copy()
                        attrs[n][s:e] += l
                n_runs += 1
    if n_runs > 0:
        ev /= n_runs
        for n in ATTR_NAMES: attrs[n] /= n_runs
    return ev, attrs, n_runs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts_orig", nargs="+", required=True)
    p.add_argument("--ckpts_eng", nargs="+", required=True)
    p.add_argument("--features_dir_orig", type=Path, default=Path("artifacts/pose_features"))
    p.add_argument("--features_dir_eng", type=Path, default=Path("artifacts/pose_features_eng"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.65)
    p.add_argument("--min_distance", type=int, default=6)
    p.add_argument("--attr_avg_half", type=int, default=4)
    p.add_argument("--weight_orig", type=float, default=1.0)
    p.add_argument("--weight_eng", type=float, default=1.0)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    models_orig = load_models(args.ckpts_orig, device)
    models_eng = load_models(args.ckpts_eng, device)
    print(f"[meta] orig {len(models_orig)} models, eng {len(models_eng)} models, tta={args.tta}")

    test_videos = pd.read_csv(TEST_VIDEOS)
    all_rows = []
    for _, vrow in test_videos.iterrows():
        vk = vrow["video_key"]
        p_orig = args.features_dir_orig / f"{vk}.npz"
        p_eng = args.features_dir_eng / f"{vk}.npz"
        if not p_orig.exists() or not p_eng.exists():
            print(f"missing feat for {vk}"); continue
        feats_orig = np.load(p_orig)["features"].astype(np.float32)
        feats_eng = np.load(p_eng)["features"].astype(np.float32)
        fps_native = float(np.load(p_orig)["fps"]) if "fps" in np.load(p_orig).files else 30.0
        T = len(feats_orig)

        ev_o, attrs_o, n_o = forward_group(models_orig, feats_orig, device, tta=args.tta)
        ev_e, attrs_e, n_e = forward_group(models_eng, feats_eng, device, tta=args.tta)

        wo = args.weight_orig; we = args.weight_eng
        total_w = wo + we
        ev = (wo * ev_o + we * ev_e) / total_w
        attrs = {n: (wo * attrs_o[n] + we * attrs_e[n]) / total_w for n in ATTR_NAMES}

        peaks = peak_pick(ev, args.threshold, args.min_distance)[:200]
        print(f"[meta] {vk}: {len(peaks)} peaks")

        h = args.attr_avg_half
        for native_frame, score in peaks:
            lo = max(0, native_frame - h); hi = min(T, native_frame + h + 1)
            attr_decided = {n: int(attrs[n][lo:hi].mean(0).argmax()) for n in ATTR_NAMES}
            t_sec = native_frame / max(fps_native, 1e-3)
            metric_frame = int(round(t_sec * 30.0))
            row = {
                "video_id": vrow["video_id"],
                "agn_index": int(vrow["agn_index"]),
                "video_key": vk,
                "frame": metric_frame,
                "score": score,
                **{n: CLASS_NAMES[n][attr_decided[n]] for n in ATTR_NAMES},
                "clear": "true",
            }
            all_rows.append(row)

    pred_df = pd.DataFrame(all_rows)
    sample = pd.read_csv(SAMPLE_SUBMISSION)
    n_required = len(sample)
    if len(pred_df) > n_required:
        pred_df = pred_df.sort_values("score", ascending=False).head(n_required)
    keep = ["video_id", "agn_index", "video_key", "frame",
            "fighter", "punch_type", "hand", "target", "effectiveness", "clear"]
    pred_df = pred_df[keep].reset_index(drop=True)
    if len(pred_df) < n_required:
        pad = sample.iloc[len(pred_df):].copy().drop(columns=["id"])
        pad["clear"] = "false"
        pred_df = pd.concat([pred_df, pad], ignore_index=True)
    pred_df.insert(0, "id", np.arange(1, len(pred_df) + 1))
    pred_df.to_csv(args.out, index=False)
    n_real = int((pred_df["clear"].astype(str).str.lower() == "true").sum())
    print(f"[meta] wrote {args.out} rows={len(pred_df)} clear_true={n_real}")


if __name__ == "__main__":
    main()
