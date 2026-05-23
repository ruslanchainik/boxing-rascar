from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .tcn_model import CLASS_NAMES


LABEL_MAPS = {name: {v: i for i, v in enumerate(values)} for name, values in CLASS_NAMES.items()}
IGNORE_INDEX = -100
ATTR_NAMES = ["fighter", "punch_type", "hand", "target", "effectiveness"]


def gaussian_kernel(half_width: int, sigma: float) -> np.ndarray:
    x = np.arange(-half_width, half_width + 1, dtype=np.float32)
    return np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)


@dataclass
class VideoData:
    video_key: str
    features: np.ndarray            # (T, F)
    event_target: np.ndarray        # (T,) gaussian heatmap [0,1]
    attr_targets: dict[str, np.ndarray]  # name -> (T,) int, IGNORE elsewhere
    attr_weights: dict[str, np.ndarray]  # name -> (T,) float, mask (1 at gt frames, 0 elsewhere)
    fps_native: float
    real_fps: float = 30.0          # for time mapping (kept; metric uses 30)


def load_video_data(
    feat_path: Path,
    punches: pd.DataFrame,
    sigma: float = 3.0,
    half_width: int = 9,
) -> VideoData:
    data = np.load(feat_path)
    features = data["features"].astype(np.float32)
    T = len(features)
    event = np.zeros(T, dtype=np.float32)
    attr_targets = {n: np.full(T, IGNORE_INDEX, dtype=np.int64) for n in ATTR_NAMES}
    attr_weights = {n: np.zeros(T, dtype=np.float32) for n in ATTR_NAMES}

    kernel = gaussian_kernel(half_width, sigma)
    # use only clear=true
    clear_mask = punches["clear"].astype(str).str.lower().eq("true")
    for _, row in punches[clear_mask].iterrows():
        f = int(row["frame"])
        if not (0 <= f < T):
            continue
        # add gaussian event peak
        lo = max(0, f - half_width)
        hi = min(T, f + half_width + 1)
        klo = lo - (f - half_width)
        khi = klo + (hi - lo)
        event[lo:hi] = np.maximum(event[lo:hi], kernel[klo:khi])
        # attribute targets only at exact peak frame
        for name in ATTR_NAMES:
            attr_targets[name][f] = LABEL_MAPS[name][str(row[name])]
            attr_weights[name][f] = 1.0

    return VideoData(
        video_key=feat_path.stem,
        features=features,
        event_target=event,
        attr_targets=attr_targets,
        attr_weights=attr_weights,
        fps_native=float(data["fps"]) if "fps" in data.files else 30.0,
    )


class PerFrameDataset(Dataset):
    """Random temporal crops over videos. Yields per-frame targets."""

    def __init__(
        self,
        videos_data: list[VideoData],
        crop_len: int = 512,
        crops_per_epoch_per_video: int = 6,
        balance_positive: bool = True,
        seed: int = 2026,
    ) -> None:
        self.videos = videos_data
        self.crop_len = crop_len
        self.crops_per_epoch_per_video = crops_per_epoch_per_video
        self.balance_positive = balance_positive
        self.rng = np.random.default_rng(seed)
        # positive frames per video for sampling around peaks
        self.pos_frames = {
            vd.video_key: np.where(vd.event_target > 0.5)[0]
            for vd in videos_data
        }

    def __len__(self) -> int:
        return len(self.videos) * self.crops_per_epoch_per_video

    def _sample_start(self, vd: VideoData) -> int:
        T = len(vd.features)
        if T <= self.crop_len:
            return 0
        positives = self.pos_frames[vd.video_key]
        if self.balance_positive and len(positives) > 0 and self.rng.random() < 0.85:
            center = int(self.rng.choice(positives))
            jitter = int(self.rng.integers(-self.crop_len // 3, self.crop_len // 3 + 1))
            start = center + jitter - self.crop_len // 2
            return int(np.clip(start, 0, T - self.crop_len))
        return int(self.rng.integers(0, T - self.crop_len + 1))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        vd = self.videos[idx % len(self.videos)]
        start = self._sample_start(vd)
        end = start + self.crop_len
        T = len(vd.features)
        feat = vd.features[start:min(end, T)]
        event = vd.event_target[start:min(end, T)]
        attrs = {n: vd.attr_targets[n][start:min(end, T)] for n in ATTR_NAMES}
        weights = {n: vd.attr_weights[n][start:min(end, T)] for n in ATTR_NAMES}
        pad = end - min(end, T)
        if pad > 0:
            feat = np.concatenate([feat, np.zeros((pad, feat.shape[1]), dtype=feat.dtype)], 0)
            event = np.concatenate([event, np.zeros(pad, dtype=event.dtype)], 0)
            for n in ATTR_NAMES:
                attrs[n] = np.concatenate([attrs[n], np.full(pad, IGNORE_INDEX, dtype=np.int64)], 0)
                weights[n] = np.concatenate([weights[n], np.zeros(pad, dtype=np.float32)], 0)
        out = {
            "features": torch.from_numpy(feat),
            "event": torch.from_numpy(event),
        }
        for n in ATTR_NAMES:
            out[f"target_{n}"] = torch.from_numpy(attrs[n])
            out[f"weight_{n}"] = torch.from_numpy(weights[n])
        return out


def collate(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], 0) for k in keys}


def class_weights_inv_freq(videos_data: list[VideoData]) -> dict[str, torch.Tensor]:
    """Inverse frequency weights per attribute, computed across all positive frames."""
    weights = {}
    for name in ATTR_NAMES:
        n_classes = len(CLASS_NAMES[name])
        counts = np.zeros(n_classes, dtype=np.float64)
        for vd in videos_data:
            tgt = vd.attr_targets[name]
            w = vd.attr_weights[name]
            mask = (tgt != IGNORE_INDEX) & (w > 0)
            for c in range(n_classes):
                counts[c] += int(((tgt == c) & mask).sum())
        total = counts.sum()
        if total == 0:
            weights[name] = torch.ones(n_classes, dtype=torch.float32)
            continue
        w = total / (n_classes * np.maximum(counts, 1.0))
        weights[name] = torch.tensor(w, dtype=torch.float32)
    return weights
