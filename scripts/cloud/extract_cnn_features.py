"""Extract per-frame CNN features (RegNetY-002 or ResNet18) from RGB crops.

Runs on DataSphere V100. Reads rgb_crops/*.mp4, outputs cnn_features/*.npz per video.

Output schema:
  features: (T_30fps, D) — D ~ 368 (RegNetY-002) or 512 (ResNet18)
  fps: 30.0 (resampled)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--crops_dir", type=Path, default=Path("input/rgb_crops"))
    p.add_argument("--out_dir", type=Path, default=Path("output/cnn_features"))
    p.add_argument("--backbone", default="regnety_002", help="timm model")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--target_fps", type=float, default=30.0,
                   help="resample features to this fps to align with pose features")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class FeatExtractor(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.model = timm.create_model(name, pretrained=True, num_classes=0, global_pool="avg")
        self.model.eval()
        cfg = self.model.default_cfg
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=cfg["mean"], std=cfg["std"]),
        ])
        with torch.no_grad():
            d = torch.zeros(1, 3, cfg["input_size"][1], cfg["input_size"][2])
            f = self.model(d)
        self.feat_dim = f.shape[1]

    def forward(self, x):
        return self.model(x)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    extractor = FeatExtractor(args.backbone).to(device).eval()
    print(f"[cnn] backbone={args.backbone} feat_dim={extractor.feat_dim} device={device}")

    files = sorted(args.crops_dir.glob("*.mp4"))
    for mp4 in files:
        vk = mp4.stem
        out_path = args.out_dir / f"{vk}.npz"
        if out_path.exists():
            print(f"skip {vk}"); continue

        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            print(f"cant open {mp4}"); continue
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 12.5)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{vk}: {n_frames} frames @ {src_fps:.2f}")

        # Read all frames into memory (small, 224x398 at 12.5 fps, ~5min video = ~3700 frames * 0.5MB = ~2GB)
        # Better stream + batch
        feats_list = []
        batch_imgs = []
        with torch.no_grad():
            while True:
                ok, fr = cap.read()
                if not ok: break
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                t = extractor.transform(rgb)
                batch_imgs.append(t)
                if len(batch_imgs) >= args.batch:
                    x = torch.stack(batch_imgs).to(device)
                    f = extractor(x).cpu().numpy()
                    feats_list.append(f)
                    batch_imgs = []
            if batch_imgs:
                x = torch.stack(batch_imgs).to(device)
                f = extractor(x).cpu().numpy()
                feats_list.append(f)
        cap.release()
        feats = np.concatenate(feats_list, axis=0).astype(np.float32) if feats_list else np.zeros((0, extractor.feat_dim), np.float32)

        # Resample to target_fps by repeating frames
        if abs(src_fps - args.target_fps) > 1e-2:
            T_src = len(feats)
            T_dst = int(round(T_src * args.target_fps / src_fps))
            idx = (np.arange(T_dst) * src_fps / args.target_fps).astype(np.int64)
            idx = np.clip(idx, 0, T_src - 1)
            feats = feats[idx]

        np.savez_compressed(out_path, features=feats, fps=args.target_fps, src_fps=src_fps)
        print(f"  saved {out_path.name}: shape={feats.shape}")


if __name__ == "__main__":
    main()
