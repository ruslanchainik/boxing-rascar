"""Grid-search threshold + min_distance over ensemble of TCN checkpoints WITH TTA color-swap."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from boxing_lstm_pipeline.tcn_dataset import ATTR_NAMES, IGNORE_INDEX, LABEL_MAPS, load_video_data
from boxing_lstm_pipeline.tcn_dataset_v2 import color_swap_features
from boxing_lstm_pipeline.tcn_model import BoxingTCN, CLASS_NAMES
from boxing_lstm_pipeline.paths import MODEL_DIR, POSE_DIR, TRAIN_PUNCHES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", type=Path, default=POSE_DIR)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def split_val(keys, val_ratio, seed):
    rng = np.random.default_rng(seed); keys = sorted(keys); idx = np.arange(len(keys)); rng.shuffle(idx)
    n_val = max(1, int(len(keys) * val_ratio))
    return {keys[i] for i in idx[:n_val]}


def peak_pick_nms(scores, threshold, min_distance):
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


def score_video(peaks, attrs, gt_frames, gt_labels):
    pred_idx = [p[0] for p in peaks]
    cands = []
    for gi, g in enumerate(gt_frames):
        for pi, p in enumerate(pred_idx):
            d = abs(p - g)
            if d <= 30: cands.append((d, gi, pi))
    cands.sort()
    mg, mp, pairs = set(), set(), []
    for d, gi, pi in cands:
        if gi in mg or pi in mp: continue
        mg.add(gi); mp.add(pi); pairs.append((gi, pi, d))
    tp = len(pairs); fp = len(pred_idx)-tp; fn = len(gt_frames)-tp
    time_errs = [min(1.0,(d/30.0)/0.5) for _,_,d in pairs]
    ac = {n:0 for n in ATTR_NAMES}; at = 0
    for gi, pi, d in pairs:
        for n in ATTR_NAMES:
            gc = gt_labels[gi][n]
            if gc == IGNORE_INDEX: continue
            pc = int(attrs[n][pred_idx[pi]].argmax())
            ac[n] += int(pc == gc)
        at += 1
    n_gt = len(gt_frames)
    st = 1 - np.mean(time_errs) if time_errs else 0.0
    accs = {n: ac[n]/max(at,1) for n in ATTR_NAMES}
    fp_pen = fp / max(n_gt + fp, 1)
    sc = ((0.5*st + 0.2*accs["fighter"] + 0.1*accs["punch_type"]
           + 0.08*accs["effectiveness"] + 0.06*accs["hand"] + 0.06*accs["target"])
          * (tp/max(n_gt,1))) - fp_pen
    return {"score": float(np.clip(sc, 0, 1)), "tp": tp, "fp": fp, "fn": fn}


def main():
    args = parse_args()
    device = args.device
    models = []
    for c in args.ckpts:
        ckpt = torch.load(c, map_location=device, weights_only=False)
        m = BoxingTCN(input_dim=ckpt["input_dim"], hidden_dim=ckpt["hidden_dim"],
                      n_blocks=ckpt["n_blocks"], dropout=ckpt.get("dropout", 0.15)).to(device)
        m.load_state_dict(ckpt["state_dict"]); m.eval(); models.append(m)
    print(f"[tune-TTA] ensemble of {len(models)}")

    punches = pd.read_csv(TRAIN_PUNCHES)
    feat_paths = sorted(args.features_dir.glob("*.npz"))
    available = {p.stem for p in feat_paths if (punches["video_key"] == p.stem).any()}
    val_keys = split_val(sorted(available), args.val_ratio, args.seed)

    cached = []
    with torch.no_grad():
        for p in feat_paths:
            if p.stem not in val_keys: continue
            vp = punches[punches["video_key"] == p.stem]
            vd = load_video_data(p, vp, sigma=3.0, half_width=9)
            T = len(vd.features); chunk = 4096
            ev = np.zeros(T, np.float32)
            attrs = {n: np.zeros((T, len(CLASS_NAMES[n])), np.float32) for n in ATTR_NAMES}

            # Two passes: original + color-swapped
            for variant_feats, is_swapped in [(vd.features, False), (color_swap_features(vd.features), True)]:
                feat = torch.from_numpy(variant_feats).unsqueeze(0).to(device)
                for s in range(0, T, chunk):
                    e = min(T, s+chunk)
                    for m in models:
                        out = m(feat[:, s:e])
                        ev[s:e] += torch.sigmoid(out["event"]).squeeze(0).cpu().numpy()
                        for n in ATTR_NAMES:
                            logits_n = out[n].squeeze(0).cpu().numpy()
                            if is_swapped and n == "fighter":
                                logits_n = logits_n[:, ::-1].copy()
                            attrs[n][s:e] += logits_n
            ev /= (2 * len(models))
            for n in ATTR_NAMES: attrs[n] /= (2 * len(models))

            gt_frames = []; gt_labels = []
            clear = vp["clear"].astype(str).str.lower() == "true"
            for _, r in vp[clear].iterrows():
                f = int(r["frame"])
                if not (0 <= f < T): continue
                gt_frames.append(f)
                gt_labels.append({n: LABEL_MAPS[n][str(r[n])] for n in ATTR_NAMES})
            cached.append((p.stem, ev, attrs, gt_frames, gt_labels))
            print(f"  cached {p.stem}: T={T} GT={len(gt_frames)}")

    thresholds = np.arange(0.20, 0.90, 0.025)
    min_distances = [6, 8, 10, 12]
    results = []
    for thr in thresholds:
        for md in min_distances:
            per_vid = []
            for vk, ev, attrs, gf, gl in cached:
                peaks = peak_pick_nms(ev, float(thr), int(md))
                r = score_video(peaks, attrs, gf, gl)
                per_vid.append(r)
            m = float(np.mean([r["score"] for r in per_vid]))
            results.append((m, float(thr), int(md),
                            sum(r["tp"] for r in per_vid),
                            sum(r["fp"] for r in per_vid),
                            sum(r["fn"] for r in per_vid)))
    results.sort(key=lambda x: -x[0])
    print("\nTop 20 (with TTA):")
    print(f"{'score':>7} {'thr':>5} {'md':>3} {'tp':>5} {'fp':>5} {'fn':>5}")
    for m, t, md, tp, fp, fn in results[:20]:
        print(f"{m:7.4f} {t:5.3f} {md:3d} {tp:5d} {fp:5d} {fn:5d}")
    best = results[0]
    print(f"\nBEST: threshold={best[1]:.3f} min_distance={best[2]} score={best[0]:.4f}")


if __name__ == "__main__":
    main()
