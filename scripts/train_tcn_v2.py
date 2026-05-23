"""Train TCN v2: anti-overfit + hard-neg + color-swap + EMA + label smoothing."""
from __future__ import annotations

import argparse
import copy
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
    ATTR_NAMES, IGNORE_INDEX, class_weights_inv_freq, collate,
)
from boxing_lstm_pipeline.tcn_dataset_v2 import (
    PerFrameDatasetV2, load_video_data_v2,
)
from boxing_lstm_pipeline.tcn_model import BoxingTCN
from boxing_lstm_pipeline.paths import MODEL_DIR, POSE_DIR, TRAIN_PUNCHES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", type=Path, default=POSE_DIR)
    p.add_argument("--out", type=Path, default=MODEL_DIR / "boxing_tcn_v2.pt")
    p.add_argument("--crop_len", type=int, default=512)
    p.add_argument("--crops_per_video", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--hidden_dim", type=int, default=192)
    p.add_argument("--n_blocks", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--half_width", type=int, default=9)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--color_swap_prob", type=float, default=0.5)
    p.add_argument("--mixup_prob", type=float, default=0.15)
    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def split_videos(keys, val_ratio, seed):
    rng = np.random.default_rng(seed)
    keys = sorted(keys); idx = np.arange(len(keys)); rng.shuffle(idx)
    n_val = max(1, int(len(keys) * val_ratio))
    val = {keys[i] for i in idx[:n_val]}
    train = {k for k in keys if k not in val}
    return train, val


def focal_bce(logits, target, gamma=2.0, pos_weight=10.0):
    p = torch.sigmoid(logits); eps = 1e-6
    pt = pos_weight * target * (1 - p) ** gamma * torch.log(p.clamp(min=eps))
    nt = (1 - target) * p ** gamma * torch.log((1 - p).clamp(min=eps))
    return -(pt + nt).mean()


def attr_loss(logits, target, weight, class_w, label_smoothing):
    mask = (target != IGNORE_INDEX) & (weight > 0)
    if not mask.any():
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(
        logits[mask], target[mask], weight=class_w, label_smoothing=label_smoothing
    )


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for s, p in zip(self.shadow.parameters(), model.parameters()):
                s.mul_(self.decay).add_(p, alpha=1 - self.decay)
            for sb, pb in zip(self.shadow.buffers(), model.buffers()):
                sb.copy_(pb)


def run_epoch(model, loader, optimizer, device, class_weights, label_smoothing, is_train, ema=None):
    model.train(is_train)
    totals = {"loss": 0.0, "event": 0.0}
    for n in ATTR_NAMES: totals[n] = 0.0
    steps = 0
    for batch in loader:
        feat = batch["features"].to(device, non_blocking=True)
        event = batch["event"].to(device, non_blocking=True)
        targets = {n: batch[f"target_{n}"].to(device) for n in ATTR_NAMES}
        weights = {n: batch[f"weight_{n}"].to(device) for n in ATTR_NAMES}
        out = model(feat)
        l_event = focal_bce(out["event"], event)
        loss = l_event
        attr_losses = {}
        mult = {"fighter": 1.2, "punch_type": 0.6, "hand": 0.5, "target": 0.5, "effectiveness": 0.5}
        for n in ATTR_NAMES:
            cw = class_weights[n].to(device) if n in {"punch_type", "effectiveness"} else None
            l = attr_loss(out[n], targets[n], weights[n], cw, label_smoothing)
            attr_losses[n] = l
            loss = loss + mult[n] * l
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)
        totals["loss"] += float(loss.item()); totals["event"] += float(l_event.item())
        for n in ATTR_NAMES: totals[n] += float(attr_losses[n].item())
        steps += 1
    return {k: v / max(steps, 1) for k, v in totals.items()}


@torch.no_grad()
def proxy_score(model, val_videos, device, threshold=0.5, min_distance=8):
    from boxing_lstm_pipeline.tcn_dataset import IGNORE_INDEX as IGN
    model.eval()
    tp=0; fp=0; fn=0
    attr_correct = {n: 0 for n in ATTR_NAMES}; attr_total = 0
    time_errs = []
    for vd in val_videos:
        feat = torch.from_numpy(vd.features).unsqueeze(0).to(device)
        out = model(feat)
        ev = torch.sigmoid(out["event"]).squeeze(0).cpu().numpy()
        T = len(ev)
        peaks = []
        for i in range(T):
            if ev[i] < threshold: continue
            lo = max(0, i-min_distance); hi = min(T, i+min_distance+1)
            if ev[i] >= ev[lo:hi].max() - 1e-6:
                peaks.append(i)
        gt_centers = sorted({int(f) for f in np.where(vd.event_target >= 0.99)[0]})
        used = set()
        for g in gt_centers:
            best = None; best_d = 31
            for j, p in enumerate(peaks):
                if j in used: continue
                d = abs(p - g)
                if d <= 30 and d < best_d: best_d = d; best = j
            if best is not None:
                used.add(best); tp += 1
                time_errs.append(best_d / 30.0)
                pf = peaks[best]
                for n in ATTR_NAMES:
                    pc = int(out[n][0, pf].argmax().item())
                    gc = int(vd.attr_targets[n][g])
                    if gc != IGN:
                        attr_correct[n] += int(pc == gc)
                attr_total += 1
            else:
                fn += 1
        fp += len(peaks) - len(used)
    n_gt = tp + fn
    score_time = 1 - np.mean([min(1.0, e/0.5) for e in time_errs]) if time_errs else 0.0
    accs = {n: attr_correct[n]/max(attr_total,1) for n in ATTR_NAMES}
    fp_pen = fp / max(n_gt + fp, 1)
    score = ((0.5*score_time + 0.2*accs["fighter"] + 0.1*accs["punch_type"]
              + 0.08*accs["effectiveness"] + 0.06*accs["hand"] + 0.06*accs["target"])
             * (tp/max(n_gt,1))) - fp_pen
    return {"score": float(np.clip(score, 0, 1)), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp/max(tp+fp,1), "recall": tp/max(tp+fn,1),
            "time_err": float(np.mean(time_errs)) if time_errs else 1.0,
            **{f"acc_{n}": accs[n] for n in ATTR_NAMES}}


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    punches = pd.read_csv(TRAIN_PUNCHES)
    feat_paths = sorted(args.features_dir.glob("*.npz"))
    available = {p.stem for p in feat_paths}
    train_keys, val_keys = split_videos(sorted(available), args.val_ratio, args.seed)
    print(f"[train_v2 seed={args.seed}] train={len(train_keys)} val={len(val_keys)}")

    train_videos, val_videos = [], []
    for p in feat_paths:
        vk = p.stem
        vp = punches[punches["video_key"] == vk]
        if len(vp) == 0: continue
        vd = load_video_data_v2(p, vp, sigma=args.sigma, half_width=args.half_width,
                                hard_neg_weight=1.0)
        if vk in train_keys: train_videos.append(vd)
        elif vk in val_keys: val_videos.append(vd)
    if not train_videos: raise SystemExit("No train videos")

    feat_dim = train_videos[0].features.shape[1]
    class_weights = class_weights_inv_freq(train_videos)

    train_ds = PerFrameDatasetV2(
        train_videos, crop_len=args.crop_len,
        crops_per_epoch_per_video=args.crops_per_video,
        balance_positive=True,
        color_swap_prob=args.color_swap_prob,
        temporal_mixup_prob=args.mixup_prob,
        seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate)
    val_ds = PerFrameDatasetV2(
        val_videos if val_videos else train_videos[:1],
        crop_len=args.crop_len, crops_per_epoch_per_video=3,
        balance_positive=False, color_swap_prob=0.0, temporal_mixup_prob=0.0,
        seed=args.seed + 1,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate)

    model = BoxingTCN(input_dim=feat_dim, hidden_dim=args.hidden_dim,
                       n_blocks=args.n_blocks, dropout=args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None

    best_score = -1.0
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, optimizer, args.device, class_weights,
                       args.label_smoothing, True, ema=ema)
        vl = run_epoch(model, val_loader, optimizer, args.device, class_weights,
                       args.label_smoothing, False)
        scheduler.step()
        print(f"epoch {epoch:02d} ({time.time()-t0:.1f}s) train_loss={tr['loss']:.4f} "
              f"val_loss={vl['loss']:.4f} event={vl['event']:.4f}", flush=True)
        # eval EMA
        if epoch % 2 == 0 or epoch == args.epochs:
            eval_model = ema.shadow if ema else model
            m = proxy_score(eval_model, val_videos, args.device, threshold=0.5, min_distance=8)
            print(f"  [val EMA] score={m['score']:.4f} P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"time_err={m['time_err']:.3f} "
                  f"f={m['acc_fighter']:.3f} type={m['acc_punch_type']:.3f} "
                  f"hand={m['acc_hand']:.3f} tgt={m['acc_target']:.3f} eff={m['acc_effectiveness']:.3f} "
                  f"(tp={m['tp']} fp={m['fp']} fn={m['fn']})", flush=True)
            save_now = False; reason = ""
            if m["score"] > best_score:
                best_score = m["score"]; save_now = True
                reason = f"score={best_score:.4f}"
            elif vl["loss"] < best_val_loss:
                best_val_loss = vl["loss"]; save_now = True
                reason = f"val_loss={best_val_loss:.4f}"
            if save_now:
                torch.save({
                    "state_dict": eval_model.state_dict(),
                    "input_dim": feat_dim, "hidden_dim": args.hidden_dim,
                    "n_blocks": args.n_blocks, "dropout": args.dropout,
                    "score": max(best_score, 0.0), "seed": args.seed,
                }, args.out)
                print(f"  saved {args.out} ({reason})", flush=True)
    if best_score < 0:
        eval_model = ema.shadow if ema else model
        torch.save({
            "state_dict": eval_model.state_dict(),
            "input_dim": feat_dim, "hidden_dim": args.hidden_dim,
            "n_blocks": args.n_blocks, "dropout": args.dropout,
            "score": 0.0, "seed": args.seed,
        }, args.out)
        print(f"  saved final {args.out}")


if __name__ == "__main__":
    main()
