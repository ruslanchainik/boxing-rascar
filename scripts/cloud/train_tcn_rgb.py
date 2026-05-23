"""Train TCN with RGB CNN features (+ optional pose features) on DataSphere.

Combines pose features (492-dim) with per-frame CNN features (e.g. 368-dim from regnety_002).
Final feature_dim ~ 860, trains in ~10-20 minutes on V100.

Inputs expected on DataSphere:
  /home/jupyter/input/pose_features_glove/<video_key>.npz  (existing 492-dim features)
  /home/jupyter/input/cnn_features/<video_key>.npz           (new CNN features)
  /home/jupyter/input/train/punches.csv
  /home/jupyter/input/train/videos.csv
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = {
    "fighter": ["red", "blue"],
    "punch_type": ["jab", "cross", "hook", "uppercut"],
    "hand": ["left", "right"],
    "target": ["head", "body"],
    "effectiveness": ["landed", "blocked", "miss"],
}
ATTR_NAMES = list(CLASS_NAMES.keys())
IGNORE_INDEX = -100
HEAD_DIMS = {"fighter": 2, "punch_type": 4, "hand": 2, "target": 2, "effectiveness": 3}


class TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float = 0.15, kernel: int = 5):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, ch)
        self.act = nn.GELU(); self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x))); h = self.drop(h)
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class BoxingTCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, n_blocks: int = 8, dropout: float = 0.2):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([TCNBlock(hidden_dim, dilation=2 ** (i % 5), dropout=dropout) for i in range(n_blocks)])
        self.event_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.attr_heads = nn.ModuleDict({name: nn.Conv1d(hidden_dim, dim, kernel_size=1) for name, dim in HEAD_DIMS.items()})

    def forward(self, x):
        x = self.input_norm(x)
        h = self.proj(x.transpose(1, 2))
        for b in self.blocks: h = b(h)
        out = {"event": self.event_head(h).squeeze(1)}
        for name, head in self.attr_heads.items():
            out[name] = head(h).transpose(1, 2)
        return out


def gaussian_kernel(half_w, sigma):
    x = np.arange(-half_w, half_w + 1, dtype=np.float32)
    return np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)


class PerFrameDataset(Dataset):
    def __init__(self, items: list[dict], crop_len: int, crops_per_video: int, seed: int):
        self.items = items
        self.crop_len = crop_len; self.crops_per_video = crops_per_video
        self.rng = np.random.default_rng(seed)

    def __len__(self): return len(self.items) * self.crops_per_video

    def __getitem__(self, idx):
        item = self.items[idx % len(self.items)]
        feat = item["features"]; ev = item["event_target"]; attrs = item["attr_targets"]; weights = item["attr_weights"]
        T = len(feat)
        # bias to positive frames
        pos = np.where(ev > 0.5)[0]
        if len(pos) and self.rng.random() < 0.85:
            c = int(self.rng.choice(pos))
            start = max(0, min(T - self.crop_len, c + int(self.rng.integers(-self.crop_len//3, self.crop_len//3+1)) - self.crop_len//2))
        else:
            start = int(self.rng.integers(0, max(1, T - self.crop_len + 1)))
        end = start + self.crop_len
        sl = slice(start, min(end, T))
        feat_s = feat[sl]; ev_s = ev[sl]
        attrs_s = {n: attrs[n][sl].copy() for n in ATTR_NAMES}
        w_s = {n: weights[n][sl].copy() for n in ATTR_NAMES}
        pad = end - min(end, T)
        if pad > 0:
            feat_s = np.concatenate([feat_s, np.zeros((pad, feat.shape[1]), feat.dtype)], 0)
            ev_s = np.concatenate([ev_s, np.zeros(pad, ev.dtype)], 0)
            for n in ATTR_NAMES:
                attrs_s[n] = np.concatenate([attrs_s[n], np.full(pad, IGNORE_INDEX, np.int64)], 0)
                w_s[n] = np.concatenate([w_s[n], np.zeros(pad, np.float32)], 0)
        out = {"features": torch.from_numpy(feat_s), "event": torch.from_numpy(ev_s)}
        for n in ATTR_NAMES:
            out[f"target_{n}"] = torch.from_numpy(attrs_s[n])
            out[f"weight_{n}"] = torch.from_numpy(w_s[n])
        return out


def load_video(pose_path: Path, cnn_path: Path, vp: pd.DataFrame, sigma=3.0, hw=9):
    pose = np.load(pose_path)["features"].astype(np.float32)
    cnn = np.load(cnn_path)["features"].astype(np.float32)
    T = min(len(pose), len(cnn))
    feats = np.concatenate([pose[:T], cnn[:T]], axis=1)
    label_maps = {n: {v: i for i, v in enumerate(CLASS_NAMES[n])} for n in ATTR_NAMES}
    event = np.zeros(T, np.float32)
    attrs = {n: np.full(T, IGNORE_INDEX, np.int64) for n in ATTR_NAMES}
    weights = {n: np.zeros(T, np.float32) for n in ATTR_NAMES}
    kernel = gaussian_kernel(hw, sigma)
    clear = vp["clear"].astype(str).str.lower() == "true"
    for _, row in vp[clear].iterrows():
        f = int(row["frame"])
        if not (0 <= f < T): continue
        lo, hi = max(0, f-hw), min(T, f+hw+1)
        klo, khi = lo - (f-hw), (lo - (f-hw)) + (hi-lo)
        event[lo:hi] = np.maximum(event[lo:hi], kernel[klo:khi])
        for n in ATTR_NAMES:
            attrs[n][f] = label_maps[n][str(row[n])]
            weights[n][f] = 1.0
    return {"features": feats, "event_target": event, "attr_targets": attrs, "attr_weights": weights}


def focal_bce(logits, target, gamma=2.0, pos_weight=10.0):
    p = torch.sigmoid(logits); eps = 1e-6
    pt = pos_weight * target * (1 - p) ** gamma * torch.log(p.clamp(min=eps))
    nt = (1 - target) * p ** gamma * torch.log((1 - p).clamp(min=eps))
    return -(pt + nt).mean()


def attr_loss(logits, target, weight, ls=0.0):
    mask = (target != IGNORE_INDEX) & (weight > 0)
    if not mask.any(): return logits.sum() * 0.0
    return nn.functional.cross_entropy(logits[mask], target[mask], label_smoothing=ls)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pose_dir", type=Path, default=Path("input/pose_features_glove"))
    p.add_argument("--cnn_dir", type=Path, default=Path("input/cnn_features"))
    p.add_argument("--punches", type=Path, default=Path("input/train/punches.csv"))
    p.add_argument("--out", type=Path, default=Path("output/best.pt"))
    p.add_argument("--crop_len", type=int, default=512)
    p.add_argument("--crops_per_video", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_blocks", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    punches = pd.read_csv(args.punches)
    items = []
    pose_paths = sorted(args.pose_dir.glob("*.npz"))
    for p in pose_paths:
        vk = p.stem
        cnn_path = args.cnn_dir / f"{vk}.npz"
        if not cnn_path.exists(): continue
        vp = punches[punches["video_key"] == vk]
        if len(vp) == 0: continue
        items.append(load_video(p, cnn_path, vp))
        print(f"loaded {vk}: T={len(items[-1]['features'])}")
    print(f"total items: {len(items)}, feat_dim={items[0]['features'].shape[1]}")

    feat_dim = items[0]["features"].shape[1]
    train_items = items[:int(len(items) * 0.85)]
    val_items = items[int(len(items) * 0.85):]
    print(f"train={len(train_items)} val={len(val_items)}")

    ds = PerFrameDataset(train_items, args.crop_len, args.crops_per_video, args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = BoxingTCN(feat_dim, args.hidden_dim, args.n_blocks, args.dropout).to(args.device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time(); model.train()
        total_loss = 0.0; n_batch = 0
        for batch in loader:
            feat = batch["features"].to(args.device); event = batch["event"].to(args.device)
            targets = {n: batch[f"target_{n}"].to(args.device) for n in ATTR_NAMES}
            weights = {n: batch[f"weight_{n}"].to(args.device) for n in ATTR_NAMES}
            out = model(feat)
            loss = focal_bce(out["event"], event)
            mult = {"fighter": 1.2, "punch_type": 0.6, "hand": 0.5, "target": 0.5, "effectiveness": 0.5}
            for n in ATTR_NAMES:
                loss = loss + mult[n] * attr_loss(out[n], targets[n], weights[n], args.label_smoothing)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()); n_batch += 1
        scheduler.step()
        train_loss = total_loss / max(n_batch, 1)

        # Validation: average loss on val items
        model.eval(); val_loss = 0.0; vc = 0
        with torch.no_grad():
            for it in val_items:
                f = torch.from_numpy(it["features"][:args.crop_len]).unsqueeze(0).to(args.device)
                e = torch.from_numpy(it["event_target"][:args.crop_len]).unsqueeze(0).to(args.device)
                out = model(f)
                vl = focal_bce(out["event"], e)
                val_loss += float(vl.item()); vc += 1
        val_loss /= max(vc, 1)
        elapsed = time.time() - t0
        print(f"epoch {epoch:02d} ({elapsed:.1f}s) train_loss={train_loss:.4f} val_event={val_loss:.4f}", flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "state_dict": model.state_dict(),
                "input_dim": feat_dim,
                "hidden_dim": args.hidden_dim,
                "n_blocks": args.n_blocks,
                "dropout": args.dropout,
                "val_loss": best_val_loss,
            }, args.out)
            print(f"  saved (val_event={best_val_loss:.4f})", flush=True)


if __name__ == "__main__":
    main()
