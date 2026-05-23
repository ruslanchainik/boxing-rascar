"""Enhanced dataset: hard-negative mining + color-swap aug + temporal mixup."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .tcn_dataset import (
    ATTR_NAMES, IGNORE_INDEX, LABEL_MAPS, gaussian_kernel, load_video_data,
    VideoData, class_weights_inv_freq, collate,
)
from .pose_features import BASE_DIM, ROLE_DIM


def color_swap_features(features: np.ndarray) -> np.ndarray:
    """Swap red↔blue role channels.
    Layouts:
      F=426: [base(142)|d1(142)|d3(142)], each 142 = red(59)+blue(59)+pair(24)
      F=464: 426 + engineered(38) = red_eng(14)+blue_eng(14)+pair_eng(10)
    """
    out = features.copy()
    T, F = out.shape
    # Swap first 426 dims in 3 blocks of 142
    base_blocks = min(3, F // 142)
    for b in range(base_blocks):
        s = b * 142
        if s + 2 * ROLE_DIM + 24 > F:
            break
        red = out[:, s:s + ROLE_DIM].copy()
        blue = out[:, s + ROLE_DIM:s + 2 * ROLE_DIM].copy()
        out[:, s:s + ROLE_DIM] = blue
        out[:, s + ROLE_DIM:s + 2 * ROLE_DIM] = red
        pair_start = s + 2 * ROLE_DIM
        pair_red_half = out[:, pair_start:pair_start + 12].copy()
        pair_blue_half = out[:, pair_start + 12:pair_start + 24].copy()
        out[:, pair_start:pair_start + 12] = pair_blue_half
        out[:, pair_start + 12:pair_start + 24] = pair_red_half
    # Engineered features at indices 426..463 if present
    if F >= 464:
        e = 426
        red_eng = out[:, e:e + 14].copy()
        blue_eng = out[:, e + 14:e + 28].copy()
        out[:, e:e + 14] = blue_eng
        out[:, e + 14:e + 28] = red_eng
        # pair (10)
        ps = e + 28
        a = out[:, ps + 2:ps + 4].copy()
        b = out[:, ps + 4:ps + 6].copy()
        out[:, ps + 2:ps + 4] = b
        out[:, ps + 4:ps + 6] = a
        c = out[:, ps + 6:ps + 8].copy()
        d = out[:, ps + 8:ps + 10].copy()
        out[:, ps + 6:ps + 8] = d
        out[:, ps + 8:ps + 10] = c
    # Glove features at 464..483 (4 wrists × 5 = 20). Order: red_L, red_R, blue_L, blue_R
    if F >= 484:
        g = 464
        red_glove = out[:, g:g + 10].copy()
        blue_glove = out[:, g + 10:g + 20].copy()
        out[:, g:g + 10] = blue_glove
        out[:, g + 10:g + 20] = red_glove
    # Glove derived at 484..491 (4 wrists × 2 = 8)
    if F >= 492:
        gd = 484
        red_gd = out[:, gd:gd + 4].copy()
        blue_gd = out[:, gd + 4:gd + 8].copy()
        out[:, gd:gd + 4] = blue_gd
        out[:, gd + 4:gd + 8] = red_gd
    return out


def load_video_data_v2(
    feat_path: Path,
    punches: pd.DataFrame,
    sigma: float = 3.0,
    half_width: int = 9,
    hard_neg_offsets: tuple = (-15, -10, 10, 15),
    hard_neg_weight: float = 0.0,
) -> VideoData:
    """Like load_video_data but also marks hard-negative frames near GT (as event=0 forced)."""
    base = load_video_data(feat_path, punches, sigma=sigma, half_width=half_width)
    T = len(base.features)
    if hard_neg_offsets and hard_neg_weight > 0:
        # carve negative penalty around GT but not at peak
        clear = punches["clear"].astype(str).str.lower() == "true"
        for _, row in punches[clear].iterrows():
            f = int(row["frame"])
            for off in hard_neg_offsets:
                t = f + off
                if 0 <= t < T and base.event_target[t] < 0.3:
                    # ensure event_target stays low at this frame even if Gaussian leaks
                    base.event_target[t] = min(base.event_target[t], 0.0)
    return base


class PerFrameDatasetV2(Dataset):
    def __init__(
        self,
        videos_data: list[VideoData],
        crop_len: int = 512,
        crops_per_epoch_per_video: int = 12,
        balance_positive: bool = True,
        color_swap_prob: float = 0.5,
        temporal_mixup_prob: float = 0.15,
        seed: int = 2026,
    ) -> None:
        self.videos = videos_data
        self.crop_len = crop_len
        self.crops_per_epoch_per_video = crops_per_epoch_per_video
        self.balance_positive = balance_positive
        self.color_swap_prob = color_swap_prob
        self.temporal_mixup_prob = temporal_mixup_prob
        self.rng = np.random.default_rng(seed)
        self.pos_frames = {
            vd.video_key: np.where(vd.event_target > 0.5)[0] for vd in videos_data
        }
        # fighter idx maps for color-swap (swap label values)
        self.swap_attr = {
            "fighter": {0: 1, 1: 0},  # red<->blue
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

    def _crop(self, vd: VideoData):
        start = self._sample_start(vd)
        end = start + self.crop_len
        T = len(vd.features)
        feat = vd.features[start:min(end, T)]
        event = vd.event_target[start:min(end, T)]
        attrs = {n: vd.attr_targets[n][start:min(end, T)].copy() for n in ATTR_NAMES}
        weights = {n: vd.attr_weights[n][start:min(end, T)].copy() for n in ATTR_NAMES}
        pad = end - min(end, T)
        if pad > 0:
            feat = np.concatenate([feat, np.zeros((pad, feat.shape[1]), dtype=feat.dtype)], 0)
            event = np.concatenate([event, np.zeros(pad, dtype=event.dtype)], 0)
            for n in ATTR_NAMES:
                attrs[n] = np.concatenate([attrs[n], np.full(pad, IGNORE_INDEX, dtype=np.int64)], 0)
                weights[n] = np.concatenate([weights[n], np.zeros(pad, dtype=np.float32)], 0)
        return feat, event, attrs, weights

    def __getitem__(self, idx: int):
        vd = self.videos[idx % len(self.videos)]
        feat, event, attrs, weights = self._crop(vd)

        # Color swap aug
        if self.rng.random() < self.color_swap_prob:
            feat = color_swap_features(feat)
            for n, mp in self.swap_attr.items():
                tgt = attrs[n]
                mask = tgt != IGNORE_INDEX
                tgt_new = tgt.copy()
                for k, v in mp.items():
                    tgt_new[mask & (tgt == k)] = v
                attrs[n] = tgt_new

        # Temporal mixup with another video crop (event labels OR-merged, attrs from dominant)
        if self.rng.random() < self.temporal_mixup_prob:
            other_idx = int(self.rng.integers(0, len(self.videos)))
            ov = self.videos[other_idx]
            o_feat, o_event, o_attrs, o_weights = self._crop(ov)
            alpha = float(self.rng.beta(0.4, 0.4))
            alpha = max(0.1, min(0.9, alpha))
            feat = alpha * feat + (1 - alpha) * o_feat
            event = np.maximum(event, o_event * (1 - alpha))
            # attrs/weights: keep dominant only

        out = {
            "features": torch.from_numpy(feat),
            "event": torch.from_numpy(event),
        }
        for n in ATTR_NAMES:
            out[f"target_{n}"] = torch.from_numpy(attrs[n])
            out[f"weight_{n}"] = torch.from_numpy(weights[n])
        return out
