"""Enhanced predict with postprocess: snap-to-wrist-velocity + fighter smoothing + hand prior."""
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
from boxing_lstm_pipeline.paths import MODEL_DIR, POSE_DIR, SAMPLE_SUBMISSION, TEST_VIDEOS


# wrist positions within base_features (per pose_features.py)
# red kpts start 0, blue kpts start 59
RED_LW_X, RED_LW_Y = 9 * 3 + 0, 9 * 3 + 1     # 27, 28
RED_RW_X, RED_RW_Y = 10 * 3 + 0, 10 * 3 + 1   # 30, 31
BLUE_LW_X = 59 + 9 * 3 + 0; BLUE_LW_Y = 59 + 9 * 3 + 1
BLUE_RW_X = 59 + 10 * 3 + 0; BLUE_RW_Y = 59 + 10 * 3 + 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", type=Path, default=POSE_DIR)
    p.add_argument("--ckpts", nargs="+", default=[str(MODEL_DIR / "boxing_tcn.pt")])
    p.add_argument("--out", type=Path, default=Path("submission_tcn_v2.csv"))
    p.add_argument("--threshold", type=float, default=0.65)
    p.add_argument("--min_distance", type=int, default=6)
    p.add_argument("--snap_window", type=int, default=5)
    p.add_argument("--no_snap", action="store_true")
    p.add_argument("--fighter_smooth", type=int, default=5)
    p.add_argument("--no_fighter_smooth", action="store_true")
    p.add_argument("--hand_prior", action="store_true", help="enable hand-prior smoothing")
    p.add_argument("--no_attr_avg", action="store_true", help="disable attr averaging")
    p.add_argument("--attr_avg_half", type=int, default=2, help="half-window for attr averaging (default 2 = 5 frames)")
    p.add_argument("--event_smooth", type=int, default=0, help="gaussian smoothing sigma for event scores (frames); 0 = off")
    p.add_argument("--tta_colorswap", action="store_true", help="run predict twice (orig + color-swapped) and average")
    p.add_argument("--rebalance_punch_type", action="store_true", help="boost rare punch_type classes")
    p.add_argument("--rebalance_effectiveness", action="store_true", help="boost rare effectiveness classes")
    p.add_argument("--per_fight_thresholds", default=None,
                   help="comma-separated thresholds per fight: 'tour1=0.65,tour2=0.65,tour3=0.65'")
    p.add_argument("--max_predictions_per_video", type=int, default=200)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def peak_pick(scores, threshold, min_distance):
    T = len(scores)
    cands = []
    for i in range(T):
        if scores[i] < threshold:
            continue
        lo = max(0, i - min_distance)
        hi = min(T, i + min_distance + 1)
        if scores[i] >= scores[lo:hi].max() - 1e-6:
            cands.append((i, float(scores[i])))
    cands.sort(key=lambda x: -x[1])
    used = np.zeros(T, dtype=bool); kept = []
    for f, s in cands:
        lo = max(0, f - min_distance); hi = min(T, f + min_distance + 1)
        if used[lo:hi].any():
            continue
        kept.append((f, s)); used[lo:hi] = True
    kept.sort(key=lambda x: x[0])
    return kept


def wrist_speed(base, fighter_idx: int) -> np.ndarray:
    """Per-frame max wrist speed for the given fighter (0=red, 1=blue)."""
    if fighter_idx == 0:
        lwx, lwy, rwx, rwy = RED_LW_X, RED_LW_Y, RED_RW_X, RED_RW_Y
    else:
        lwx, lwy, rwx, rwy = BLUE_LW_X, BLUE_LW_Y, BLUE_RW_X, BLUE_RW_Y
    T = len(base)
    lw = base[:, [lwx, lwy]]
    rw = base[:, [rwx, rwy]]
    dlw = np.zeros(T, dtype=np.float32)
    drw = np.zeros(T, dtype=np.float32)
    dlw[1:] = np.linalg.norm(lw[1:] - lw[:-1], axis=1)
    drw[1:] = np.linalg.norm(rw[1:] - rw[:-1], axis=1)
    return np.maximum(dlw, drw)


def snap_to_velocity(frame: int, speed: np.ndarray, window: int) -> int:
    T = len(speed)
    lo = max(0, frame - window)
    hi = min(T, frame + window + 1)
    sub = speed[lo:hi]
    if sub.size == 0:
        return frame
    return lo + int(np.argmax(sub))


def smooth_fighter(fighter_logits: np.ndarray, frame: int, window: int) -> int:
    T = len(fighter_logits)
    lo = max(0, frame - window); hi = min(T, frame + window + 1)
    avg = fighter_logits[lo:hi].mean(0)
    return int(np.argmax(avg))


def majority_hand_per_fighter(rows: list[dict]) -> dict[int, int]:
    """Return majority hand idx per fighter idx (0=red,1=blue)."""
    from collections import Counter
    out = {}
    for fi in (0, 1):
        cnt = Counter([r["hand_idx"] for r in rows if r["fighter_idx"] == fi])
        out[fi] = cnt.most_common(1)[0][0] if cnt else 0
    return out


def main():
    args = parse_args()
    device = args.device

    # Load all checkpoints for ensemble
    models = []
    for ckpt_path in args.ckpts:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m = BoxingTCN(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt["hidden_dim"],
            n_blocks=ckpt["n_blocks"],
            dropout=ckpt.get("dropout", 0.15),
        ).to(device)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models.append(m)
        print(f"[predict] loaded {ckpt_path} score={ckpt.get('score','n/a')}")
    print(f"[predict] ensemble of {len(models)} model(s), threshold={args.threshold} min_distance={args.min_distance}")

    test_videos = pd.read_csv(TEST_VIDEOS)
    all_rows = []

    for _, vrow in test_videos.iterrows():
        vk = vrow["video_key"]
        fpath = args.features_dir / f"{vk}.npz"
        if not fpath.exists():
            print(f"[predict] missing {vk}"); continue
        data = np.load(fpath)
        feats = data["features"].astype(np.float32)
        base = data["base_features"].astype(np.float32) if "base_features" in data.files else feats[:, :142]
        fps_native = float(data["fps"]) if "fps" in data.files else 30.0
        T = len(feats)

        # Ensemble forward (+ optional TTA color-swap)
        # NOTE: color-swap reorders red/blue role channels; for matched fighter logits we swap back after
        from boxing_lstm_pipeline.tcn_dataset_v2 import color_swap_features

        feature_variants = [(feats, False)]
        if args.tta_colorswap:
            feature_variants.append((color_swap_features(feats), True))

        event_accum = np.zeros(T, np.float32)
        attr_accum = {n: np.zeros((T, len(CLASS_NAMES[n])), np.float32) for n in ATTR_NAMES}
        n_runs = 0
        with torch.no_grad():
            for var_feats, is_swapped in feature_variants:
                for model in models:
                    chunk = 4096
                    for s in range(0, T, chunk):
                        e = min(T, s + chunk)
                        x = torch.from_numpy(var_feats[s:e]).unsqueeze(0).to(device)
                        out = model(x)
                        event_accum[s:e] += torch.sigmoid(out["event"]).squeeze(0).cpu().numpy()
                        for n in ATTR_NAMES:
                            logits_n = out[n].squeeze(0).cpu().numpy()
                            if is_swapped and n == "fighter":
                                logits_n = logits_n[:, ::-1].copy()  # red<->blue swap back
                            attr_accum[n][s:e] += logits_n
                    n_runs += 1
        event_scores = event_accum / max(n_runs, 1)
        attr_logits = {n: attr_accum[n] / max(n_runs, 1) for n in ATTR_NAMES}

        # Class rebalance: subtract log-frequency to boost rare classes (matches inverse-freq weights of metric)
        if args.rebalance_punch_type:
            # train frequencies: jab~30%, cross~28%, hook~30%, uppercut~12% -> inv-freq weights
            inv = np.array([1.11, 1.19, 1.11, 2.78], dtype=np.float32)
            attr_logits["punch_type"] = attr_logits["punch_type"] + np.log(inv)
        if args.rebalance_effectiveness:
            # train frequencies: landed~25%, blocked~25%, miss~50% -> boost landed, blocked
            inv = np.array([1.33, 1.33, 0.67], dtype=np.float32)
            attr_logits["effectiveness"] = attr_logits["effectiveness"] + np.log(inv)

        # Optional Gaussian temporal smoothing of event scores before peak-picking
        if args.event_smooth > 0:
            sig = float(args.event_smooth)
            r = int(max(1, round(3 * sig)))
            ker = np.exp(-0.5 * ((np.arange(-r, r + 1)) / sig) ** 2)
            ker = (ker / ker.sum()).astype(np.float32)
            event_scores = np.convolve(event_scores, ker, mode="same").astype(np.float32)

        # Resolve per-fight threshold
        thr_local = args.threshold
        if args.per_fight_thresholds:
            kv = dict(p.split("=") for p in args.per_fight_thresholds.split(","))
            # Mapping by video_key prefix or by fight ID
            agn_idx = int(vrow["agn_index"])
            if 37 <= agn_idx <= 39:
                key = "tour1"
            elif 47 <= agn_idx <= 49:
                key = "tour2"
            elif 62 <= agn_idx <= 64:
                key = "tour3"
            else:
                key = "tour1"
            if key in kv:
                thr_local = float(kv[key])

        peaks = peak_pick(event_scores, thr_local, args.min_distance)
        peaks = peaks[: args.max_predictions_per_video]
        print(f"[predict] {vk}: {len(peaks)} peaks (T={T}, fps={fps_native:.2f}, thr={thr_local:.3f})")

        # Precompute per-fighter wrist speed
        speed_red = wrist_speed(base, 0)
        speed_blue = wrist_speed(base, 1)

        rows_local = []
        for native_frame, score in peaks:
            # Fighter
            if args.no_fighter_smooth:
                fi = int(attr_logits["fighter"][native_frame].argmax())
            else:
                fi = smooth_fighter(attr_logits["fighter"], native_frame, args.fighter_smooth)
            # Snap to velocity
            if args.no_snap:
                snapped = native_frame
            else:
                speed = speed_red if fi == 0 else speed_blue
                snapped = snap_to_velocity(native_frame, speed, args.snap_window)
            # Per-attribute argmax using smoothed window
            attrs = {}
            for n in ATTR_NAMES:
                if n == "fighter":
                    attrs[n] = fi
                else:
                    if args.no_attr_avg:
                        attrs[n] = int(attr_logits[n][snapped].argmax())
                    else:
                        h = max(0, int(args.attr_avg_half))
                        lo = max(0, snapped - h); hi = min(T, snapped + h + 1)
                        attrs[n] = int(attr_logits[n][lo:hi].mean(0).argmax())
            row = {
                "video_id": vrow["video_id"],
                "agn_index": int(vrow["agn_index"]),
                "video_key": vk,
                "_native_frame": snapped,
                "fps_native": fps_native,
                "score": score,
                "fighter_idx": attrs["fighter"],
                "fighter": CLASS_NAMES["fighter"][attrs["fighter"]],
                "punch_type": CLASS_NAMES["punch_type"][attrs["punch_type"]],
                "hand": CLASS_NAMES["hand"][attrs["hand"]],
                "hand_idx": attrs["hand"],
                "target": CLASS_NAMES["target"][attrs["target"]],
                "effectiveness": CLASS_NAMES["effectiveness"][attrs["effectiveness"]],
            }
            rows_local.append(row)

        # Hand-prior: soft bias toward majority hand per fighter (only when raw confidence is low)
        if args.hand_prior and rows_local:
            maj = majority_hand_per_fighter(rows_local)
            # Apply only if margin is weak — recompute hand with prior
            for r in rows_local:
                fr = r["_native_frame"]
                lo = max(0, fr - 2); hi = min(T, fr + 3)
                hand_avg = attr_logits["hand"][lo:hi].mean(0)
                # softmax then add prior
                e = np.exp(hand_avg - hand_avg.max())
                probs = e / e.sum()
                margin = abs(probs[0] - probs[1])
                if margin < 0.25:  # low confidence → trust prior
                    r["hand_idx"] = maj[r["fighter_idx"]]
                    r["hand"] = CLASS_NAMES["hand"][r["hand_idx"]]

        # Map to 30 fps frame
        for r in rows_local:
            r["frame"] = int(round(r["_native_frame"] / max(r["fps_native"], 1e-3) * 30.0))
        all_rows.extend(rows_local)

    pred_df = pd.DataFrame(all_rows)
    pred_df["clear"] = "true"

    sample = pd.read_csv(SAMPLE_SUBMISSION)
    n_required = len(sample)
    if len(pred_df) > n_required:
        pred_df = pred_df.sort_values("score", ascending=False).head(n_required)
    keep_cols = ["video_id", "agn_index", "video_key", "frame",
                 "fighter", "punch_type", "hand", "target", "effectiveness", "clear"]
    pred_df = pred_df[keep_cols].reset_index(drop=True)
    pad_rows = n_required - len(pred_df)
    if pad_rows > 0:
        pad_df = sample.iloc[len(pred_df):].copy().drop(columns=["id"])
        pad_df["clear"] = "false"
        pred_df = pd.concat([pred_df, pad_df], ignore_index=True)
    pred_df.insert(0, "id", np.arange(1, len(pred_df) + 1))
    pred_df.to_csv(args.out, index=False)
    n_real = int((pred_df["clear"].astype(str).str.lower() == "true").sum())
    print(f"[predict] wrote {args.out} rows={len(pred_df)} clear_true={n_real}")


if __name__ == "__main__":
    main()
