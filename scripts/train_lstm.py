from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

sys.path.append(str(Path(__file__).resolve().parents[1]))

from boxing_lstm_pipeline.dataset import PunchWindowDataset, collate_batch
from boxing_lstm_pipeline.model import BoxingLSTM
from boxing_lstm_pipeline.paths import MODEL_DIR, POSE_DIR, TRAIN_PUNCHES, TRAIN_VIDEOS
from boxing_lstm_pipeline.pose_features import FEATURE_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=Path, default=POSE_DIR)
    parser.add_argument("--out", type=Path, default=MODEL_DIR / "boxing_lstm.pt")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--negatives_per_positive", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def batch_to_device(batch, device: str):
    x, y = batch
    return x.to(device), {k: v.to(device) for k, v in y.items()}


def compute_loss(logits: dict[str, torch.Tensor], y: dict[str, torch.Tensor]) -> torch.Tensor:
    event_loss = nn.functional.binary_cross_entropy_with_logits(
        logits["event"].squeeze(1),
        y["event"],
        pos_weight=torch.tensor(2.0, device=y["event"].device),
    )
    loss = event_loss
    positive = y["event"] > 0.5
    if positive.any():
        for name in ["fighter", "punch_type", "hand", "target", "effectiveness"]:
            loss = loss + 0.25 * nn.functional.cross_entropy(
                logits[name][positive],
                y[name][positive],
            )
    return loss


@torch.no_grad()
def evaluate(model: BoxingLSTM, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    event_correct = 0
    attr_correct = {name: 0 for name in ["fighter", "punch_type", "hand", "target", "effectiveness"]}
    attr_total = 0
    for batch in loader:
        x, y = batch_to_device(batch, device)
        logits = model(x)
        loss = compute_loss(logits, y)
        bs = len(x)
        total_loss += float(loss.item()) * bs
        total += bs
        event_pred = (torch.sigmoid(logits["event"].squeeze(1)) >= 0.5).float()
        event_correct += int((event_pred == y["event"]).sum().item())
        positive = y["event"] > 0.5
        attr_total += int(positive.sum().item())
        if positive.any():
            for name in attr_correct:
                attr_correct[name] += int((logits[name][positive].argmax(1) == y[name][positive]).sum().item())

    metrics = {
        "loss": total_loss / max(total, 1),
        "event_acc": event_correct / max(total, 1),
    }
    for name, value in attr_correct.items():
        metrics[f"{name}_acc"] = value / max(attr_total, 1)
    return metrics


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    punches = pd.read_csv(TRAIN_PUNCHES)
    videos = pd.read_csv(TRAIN_VIDEOS)
    dataset = PunchWindowDataset(
        args.features_dir,
        punches,
        videos,
        window=args.window,
        negatives_per_positive=args.negatives_per_positive,
    )
    if len(dataset) == 0:
        raise SystemExit(f"No samples. Run scripts/extract_pose_features.py first into {args.features_dir}")

    val_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(2026))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    model = BoxingLSTM(FEATURE_DIM, hidden_dim=args.hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            x, y = batch_to_device(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = compute_loss(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item()) * len(x)
            seen += len(x)

        metrics = evaluate(model, val_loader, args.device)
        print(
            f"epoch={epoch:02d} train_loss={running/max(seen,1):.4f} "
            + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_dim": FEATURE_DIM,
                    "window": args.window,
                    "hidden_dim": args.hidden_dim,
                },
                args.out,
            )
            print(f"saved {args.out}")


if __name__ == "__main__":
    main()
