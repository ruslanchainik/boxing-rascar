from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from boxing_lstm_pipeline.tcn_dataset import (
    ATTR_NAMES,
    IGNORE_INDEX,
    PerFrameDataset,
    class_weights_inv_freq,
    collate,
    load_video_data,
)
from boxing_lstm_pipeline.tcn_model import BoxingTCN, HEAD_DIMS
from boxing_lstm_pipeline.paths import MODEL_DIR, POSE_DIR, TRAIN_PUNCHES, TRAIN_VIDEOS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", type=Path, default=POSE_DIR)
    p.add_argument("--out", type=Path, default=MODEL_DIR / "boxing_tcn.pt")
    p.add_argument("--crop_len", type=int, default=512)
    p.add_argument("--crops_per_video", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden_dim", type=int, default=192)
    p.add_argument("--n_blocks", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--half_width", type=int, default=9)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def split_videos(video_keys: list[str], val_ratio: float, seed: int) -> tuple[set, set]:
    rng = np.random.default_rng(seed)
    keys = sorted(video_keys)
    idx = np.arange(len(keys))
    rng.shuffle(idx)
    n_val = max(1, int(len(keys) * val_ratio))
    val = {keys[i] for i in idx[:n_val]}
    train = {k for k in keys if k not in val}
    return train, val


def focal_bce(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, pos_weight: float = 10.0) -> torch.Tensor:
    # logits/target: (B,T). target is gaussian-soft in [0,1].
    p = torch.sigmoid(logits)
    eps = 1e-6
    pos_term = -pos_weight * target * (1 - p) ** gamma * torch.log(p.clamp(min=eps))
    neg_term = -(1 - target) * p ** gamma * torch.log((1 - p).clamp(min=eps))
    return (pos_term + neg_term).mean()


def attr_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, class_w: torch.Tensor | None) -> torch.Tensor:
    # logits: (B,T,C); target: (B,T) long with IGNORE; weight: (B,T)
    B, T, C = logits.shape
    mask = (target != IGNORE_INDEX) & (weight > 0)
    if not mask.any():
        return logits.sum() * 0.0
    flat_logits = logits[mask]
    flat_target = target[mask]
    return nn.functional.cross_entropy(flat_logits, flat_target, weight=class_w)


def run_epoch(model, loader, optimizer, device, class_weights, is_train: bool):
    model.train(is_train)
    total = {"loss": 0.0, "event": 0.0}
    for n in ATTR_NAMES:
        total[n] = 0.0
    n_steps = 0
    for batch in loader:
        feat = batch["features"].to(device, non_blocking=True)
        event = batch["event"].to(device, non_blocking=True)
        targets = {n: batch[f"target_{n}"].to(device, non_blocking=True) for n in ATTR_NAMES}
        weights = {n: batch[f"weight_{n}"].to(device, non_blocking=True) for n in ATTR_NAMES}

        out = model(feat)
        l_event = focal_bce(out["event"], event)
        loss = l_event
        attr_losses = {}
        for n in ATTR_NAMES:
            cw = class_weights[n].to(device) if n in {"punch_type", "effectiveness"} else None
            l = attr_loss(out[n], targets[n], weights[n], cw)
            attr_losses[n] = l
            # weight by metric importance
            mult = {"fighter": 1.2, "punch_type": 0.6, "hand": 0.5, "target": 0.5, "effectiveness": 0.5}[n]
            loss = loss + mult * l

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total["loss"] += float(loss.item())
        total["event"] += float(l_event.item())
        for n in ATTR_NAMES:
            total[n] += float(attr_losses[n].item())
        n_steps += 1

    return {k: v / max(n_steps, 1) for k, v in total.items()}


@torch.no_grad()
def evaluate_metric(model, val_videos, device, threshold: float = 0.4, min_distance: int = 10) -> dict[str, float]:
    """Quick proxy validation: per-frame event AP-like and attr accuracy at GT frames."""
    model.eval()
    tp = 0; fp = 0; fn = 0
    attr_correct = {n: 0 for n in ATTR_NAMES}
    attr_total = 0
    time_errors = []
    for vd in val_videos:
        feat = torch.from_numpy(vd.features).unsqueeze(0).to(device)
        out = model(feat)
        ev = torch.sigmoid(out["event"]).squeeze(0).cpu().numpy()
        # local-maxima peak picking
        peaks = []
        T = len(ev)
        for i in range(T):
            if ev[i] < threshold:
                continue
            lo = max(0, i - min_distance)
            hi = min(T, i + min_distance + 1)
            if ev[i] >= ev[lo:hi].max() - 1e-6:
                peaks.append(i)
        gt_frames = np.where(vd.event_target > 0.5)[0]
        # find gt centers (peak of gaussian = local max)
        gt_centers = []
        for i in range(len(gt_frames)):
            f = int(gt_frames[i])
            if vd.event_target[f] >= 0.99:
                gt_centers.append(f)
        gt_centers = sorted(set(gt_centers))
        matched_pred = set()
        for g in gt_centers:
            best = None; best_d = 31  # 30 fps * 1 sec
            for j, p in enumerate(peaks):
                if j in matched_pred:
                    continue
                d = abs(p - g)
                if d <= 30 and d < best_d:
                    best_d = d; best = j
            if best is not None:
                matched_pred.add(best)
                tp += 1
                time_errors.append(best_d / 30.0)
                # attr at pred frame
                pred_frame = peaks[best]
                for n in ATTR_NAMES:
                    pred_cls = int(out[n][0, pred_frame].argmax().item())
                    gt_cls = int(vd.attr_targets[n][g])
                    if gt_cls != IGNORE_INDEX:
                        attr_correct[n] += int(pred_cls == gt_cls)
                attr_total += 1
            else:
                fn += 1
        fp += len(peaks) - len(matched_pred)

    metrics = {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "mean_time_err": float(np.mean(time_errors)) if time_errors else 1.0,
    }
    for n in ATTR_NAMES:
        metrics[f"acc_{n}"] = attr_correct[n] / max(attr_total, 1)
    # rough proxy score (like metric)
    score_time = 1 - np.mean([min(1.0, e / 0.5) for e in time_errors]) if time_errors else 0.0
    n_gt = tp + fn
    n_fp = fp
    fp_pen = n_fp / max(n_gt + n_fp, 1)
    score = (
        0.5 * score_time
        + 0.2 * metrics["acc_fighter"]
        + 0.1 * metrics["acc_punch_type"]
        + 0.08 * metrics["acc_effectiveness"]
        + 0.06 * metrics["acc_hand"]
        + 0.06 * metrics["acc_target"]
    ) * (tp / max(n_gt, 1)) - fp_pen
    metrics["score"] = float(np.clip(score, 0, 1))
    return metrics


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    punches = pd.read_csv(TRAIN_PUNCHES)
    feat_paths = sorted(args.features_dir.glob("*.npz"))
    available = {p.stem for p in feat_paths}
    print(f"[train] {len(available)} feature files available")

    # split videos by key
    train_keys, val_keys = split_videos(sorted(available), args.val_ratio, args.seed)
    print(f"[train] train={len(train_keys)} val={len(val_keys)}")

    train_videos, val_videos = [], []
    for p in feat_paths:
        vk = p.stem
        vp = punches[punches["video_key"] == vk]
        if len(vp) == 0:
            continue
        vd = load_video_data(p, vp, sigma=args.sigma, half_width=args.half_width)
        if vk in train_keys:
            train_videos.append(vd)
        elif vk in val_keys:
            val_videos.append(vd)
    print(f"[train] loaded train_videos={len(train_videos)} val_videos={len(val_videos)}")
    if not train_videos:
        raise SystemExit("No train videos with features+labels")

    feat_dim = train_videos[0].features.shape[1]
    class_weights = class_weights_inv_freq(train_videos)
    for n, w in class_weights.items():
        print(f"  class_w[{n}] = {w.tolist()}")

    train_ds = PerFrameDataset(
        train_videos,
        crop_len=args.crop_len,
        crops_per_epoch_per_video=args.crops_per_video,
        balance_positive=True,
        seed=args.seed,
    )
    val_ds = PerFrameDataset(
        val_videos if val_videos else train_videos[:1],
        crop_len=args.crop_len,
        crops_per_epoch_per_video=3,
        balance_positive=False,
        seed=args.seed + 1,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )

    device = args.device
    model = BoxingTCN(
        input_dim=feat_dim,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params/1e6:.2f}M; feat_dim={feat_dim}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_score = -1.0
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_log = run_epoch(model, train_loader, optimizer, device, class_weights, is_train=True)
        val_log = run_epoch(model, val_loader, optimizer, device, class_weights, is_train=False)
        scheduler.step()
        msg = (
            f"epoch {epoch:02d} ({time.time()-t0:.1f}s) "
            f"train_loss={train_log['loss']:.4f} val_loss={val_log['loss']:.4f} "
            f"event={val_log['event']:.4f}"
        )
        print(msg, flush=True)

        if val_videos and (epoch % 2 == 0 or epoch == args.epochs):
            m = evaluate_metric(model, val_videos, device)
            print(
                f"  [val] score={m['score']:.4f} P={m['precision']:.3f} R={m['recall']:.3f} "
                f"time_err={m['mean_time_err']:.3f} "
                f"f={m['acc_fighter']:.3f} type={m['acc_punch_type']:.3f} hand={m['acc_hand']:.3f} "
                f"tgt={m['acc_target']:.3f} eff={m['acc_effectiveness']:.3f} "
                f"(tp={m['tp']} fp={m['fp']} fn={m['fn']})",
                flush=True,
            )
            # Save when score improves OR when val_loss improves (track both independently)
            save_now = False
            reason = ""
            if m["score"] > best_score:
                best_score = m["score"]
                save_now = True
                reason = f"score={best_score:.4f}"
            if val_log["loss"] < best_val_loss:
                best_val_loss = val_log["loss"]
                if not save_now:
                    save_now = True
                    reason = f"val_loss={best_val_loss:.4f}"
            if save_now:
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "input_dim": feat_dim,
                        "hidden_dim": args.hidden_dim,
                        "n_blocks": args.n_blocks,
                        "dropout": args.dropout,
                        "score": max(best_score, 0.0),
                    },
                    args.out,
                )
                print(f"  saved {args.out} ({reason})", flush=True)

    # final save fallback
    if best_score < 0:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": feat_dim,
                "hidden_dim": args.hidden_dim,
                "n_blocks": args.n_blocks,
                "dropout": args.dropout,
                "score": 0.0,
            },
            args.out,
        )
        print(f"  saved final {args.out}")


if __name__ == "__main__":
    main()
