from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .model import CLASS_NAMES


LABEL_MAPS = {
    name: {value: i for i, value in enumerate(values)}
    for name, values in CLASS_NAMES.items()
}


@dataclass
class SampleRef:
    video_key: str
    center: int
    punch_index: int


def load_feature_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["frames"].astype(np.int32), data["features"].astype(np.float32)


def window_at(features: np.ndarray, center: int, window: int) -> np.ndarray:
    half = window // 2
    start = center - half
    end = start + window
    left_pad = max(0, -start)
    right_pad = max(0, end - len(features))
    start = max(0, start)
    end = min(len(features), end)
    x = features[start:end]
    if left_pad or right_pad:
        x = np.pad(x, ((left_pad, right_pad), (0, 0)), mode="edge")
    return x.astype(np.float32)


class PunchWindowDataset(Dataset):
    def __init__(
        self,
        features_dir: Path,
        punches: pd.DataFrame,
        videos: pd.DataFrame,
        window: int = 64,
        negatives_per_positive: int = 2,
        min_neg_distance: int = 45,
        seed: int = 2026,
    ) -> None:
        self.features_dir = Path(features_dir)
        self.window = window
        self.rng = np.random.default_rng(seed)
        self.features: dict[str, np.ndarray] = {}
        self.samples: list[SampleRef] = []
        self.labels: list[dict[str, int | float]] = []

        punches = punches[punches["clear"].astype(str).str.lower().eq("true")].copy()
        video_frames = videos.set_index("video_key")["frame_count"].to_dict()

        for video_key, group in punches.groupby("video_key"):
            path = self.features_dir / f"{video_key}.npz"
            if not path.exists():
                continue
            _, feats = load_feature_npz(path)
            self.features[video_key] = feats
            positives = sorted(set(int(f) for f in group["frame"].tolist()))

            for idx, row in group.iterrows():
                center = int(row["frame"])
                self.samples.append(SampleRef(video_key, center, int(idx)))
                self.labels.append(self.encode_positive(row))

            n_frames = int(min(len(feats), video_frames.get(video_key, len(feats))))
            if n_frames <= 0:
                continue
            for _ in range(max(1, len(group) * negatives_per_positive)):
                center = self.sample_negative_center(n_frames, positives, min_neg_distance)
                self.samples.append(SampleRef(video_key, center, -1))
                self.labels.append(self.encode_negative())

    def sample_negative_center(self, n_frames: int, positives: list[int], min_distance: int) -> int:
        for _ in range(100):
            center = int(self.rng.integers(0, n_frames))
            if all(abs(center - p) >= min_distance for p in positives):
                return center
        return int(self.rng.integers(0, n_frames))

    @staticmethod
    def encode_positive(row: pd.Series) -> dict[str, int | float]:
        label: dict[str, int | float] = {"event": 1.0}
        for name, mapping in LABEL_MAPS.items():
            label[name] = mapping[str(row[name])]
        return label

    @staticmethod
    def encode_negative() -> dict[str, int | float]:
        label: dict[str, int | float] = {"event": 0.0}
        for name in LABEL_MAPS:
            label[name] = -100
        return label

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ref = self.samples[idx]
        x = window_at(self.features[ref.video_key], ref.center, self.window)
        labels = self.labels[idx]
        y = {
            "event": torch.tensor(labels["event"], dtype=torch.float32),
        }
        for name in LABEL_MAPS:
            y[name] = torch.tensor(labels[name], dtype=torch.long)
        return torch.from_numpy(x), y


def collate_batch(batch):
    xs, ys = zip(*batch)
    out_y = {}
    for key in ys[0]:
        out_y[key] = torch.stack([y[key] for y in ys])
    return torch.stack(xs), out_y

