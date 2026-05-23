from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .paths import ROOT


N_KPTS = 17
KPT_DIM = N_KPTS * 3
ROLE_DIM = KPT_DIM + 4 + 3 + 1
PAIR_DIM = 24
BASE_DIM = ROLE_DIM * 2 + PAIR_DIM
FEATURE_DIM = BASE_DIM * 3

RED1_LOWER = np.array([0, 90, 60])
RED1_UPPER = np.array([14, 255, 255])
RED2_LOWER = np.array([165, 90, 60])
RED2_UPPER = np.array([180, 255, 255])
BLUE_LOWER = np.array([92, 80, 50])
BLUE_UPPER = np.array([132, 255, 255])
WHITE_LOWER = np.array([0, 0, 145])
WHITE_UPPER = np.array([180, 70, 255])

NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12


@dataclass
class PersonObs:
    tid: int
    bbox: np.ndarray
    kpts: np.ndarray
    colors: np.ndarray
    center: np.ndarray


def video_path_from_manifest(row: pd.Series) -> Path:
    return ROOT / str(row["video_path"])


def safe_point(kpts: np.ndarray, idx: int, min_conf: float = 0.1) -> np.ndarray | None:
    if idx >= len(kpts) or kpts[idx, 2] < min_conf:
        return None
    return kpts[idx, :2].astype(np.float32)


def mean_point(kpts: np.ndarray, idxs: list[int], min_conf: float = 0.1) -> np.ndarray | None:
    pts = [safe_point(kpts, i, min_conf) for i in idxs]
    pts = [p for p in pts if p is not None]
    if not pts:
        return None
    return np.mean(np.stack(pts), axis=0)


def add_circle(mask: np.ndarray, center: np.ndarray | None, radius: int) -> None:
    if center is None:
        return
    cv2.circle(mask, tuple(np.round(center).astype(int)), max(2, radius), 255, -1)


def add_poly(mask: np.ndarray, pts: list[np.ndarray | None]) -> None:
    pts = [p for p in pts if p is not None]
    if len(pts) < 3:
        return
    cv2.fillConvexPoly(mask, np.round(np.stack(pts)).astype(np.int32), 255)


COLOR_DOWNSAMPLE = 4  # process color at 1/4 resolution; pose still full-res


def color_ratios(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n = int(np.count_nonzero(mask))
    if n <= 10:
        return np.zeros(3, dtype=np.float32)

    if COLOR_DOWNSAMPLE > 1:
        h, w = frame.shape[:2]
        nh, nw = h // COLOR_DOWNSAMPLE, w // COLOR_DOWNSAMPLE
        frame_small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        mask_small = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        n_small = int(np.count_nonzero(mask_small))
        if n_small <= 4:
            return np.zeros(3, dtype=np.float32)
        frame, mask, n = frame_small, mask_small, n_small

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, RED1_LOWER, RED1_UPPER),
        cv2.inRange(hsv, RED2_LOWER, RED2_UPPER),
    )
    blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    white_mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)

    return np.array(
        [
            np.count_nonzero(cv2.bitwise_and(red_mask, red_mask, mask=mask)) / n,
            np.count_nonzero(cv2.bitwise_and(blue_mask, blue_mask, mask=mask)) / n,
            np.count_nonzero(cv2.bitwise_and(white_mask, white_mask, mask=mask)) / n,
        ],
        dtype=np.float32,
    )


def person_color_features(frame: np.ndarray, kpts: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = bbox.astype(float)
    radius = int(max(5.0, 0.035 * max(x2 - x1, y2 - y1)))

    ls, rs = safe_point(kpts, L_SHOULDER), safe_point(kpts, R_SHOULDER)
    lh, rh = safe_point(kpts, L_HIP), safe_point(kpts, R_HIP)
    lw, rw = safe_point(kpts, L_WRIST), safe_point(kpts, R_WRIST)
    le, re = safe_point(kpts, L_ELBOW), safe_point(kpts, R_ELBOW)

    add_poly(mask, [ls, rs, rh, lh])
    add_poly(mask, [lh, rh, (rh + np.array([0, radius * 3])) if rh is not None else None,
                    (lh + np.array([0, radius * 3])) if lh is not None else None])
    add_circle(mask, lw, radius * 2)
    add_circle(mask, rw, radius * 2)
    add_circle(mask, le, radius)
    add_circle(mask, re, radius)

    if np.count_nonzero(mask) <= 10:
        cx1 = int(np.clip(x1 + 0.25 * (x2 - x1), 0, w - 1))
        cx2 = int(np.clip(x1 + 0.75 * (x2 - x1), 0, w - 1))
        cy1 = int(np.clip(y1 + 0.20 * (y2 - y1), 0, h - 1))
        cy2 = int(np.clip(y1 + 0.70 * (y2 - y1), 0, h - 1))
        mask[cy1:cy2, cx1:cx2] = 255

    return color_ratios(frame, mask)


class RoleAssigner:
    def __init__(self) -> None:
        self.prev_centers: dict[str, np.ndarray | None] = {"red": None, "blue": None}

    def update(
        self,
        frame: np.ndarray,
        track_ids: np.ndarray,
        bboxes: np.ndarray,
        keypoints: np.ndarray,
    ) -> dict[str, PersonObs]:
        h, w = frame.shape[:2]
        diag = float(np.hypot(w, h))
        candidates: list[PersonObs] = []

        for tid, bbox, kpts in zip(track_ids, bboxes, keypoints):
            x1, y1, x2, y2 = bbox.astype(np.float32)
            if (y2 - y1) < 0.16 * h:
                continue
            colors = person_color_features(frame, kpts, bbox)
            red, blue, white = colors
            if white > 0.16 and max(red, blue) < 0.08:
                continue
            candidates.append(
                PersonObs(
                    tid=int(tid),
                    bbox=bbox.astype(np.float32),
                    kpts=kpts.astype(np.float32),
                    colors=colors,
                    center=np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32),
                )
            )

        if not candidates:
            self.prev_centers = {"red": None, "blue": None}
            return {}

        cost = np.ones((len(candidates), 2), dtype=np.float32)
        for i, cand in enumerate(candidates):
            color_scores = {"red": float(cand.colors[0]), "blue": float(cand.colors[1])}
            for j, role in enumerate(("red", "blue")):
                score = color_scores[role]
                prev = self.prev_centers[role]
                if prev is not None:
                    dist = float(np.linalg.norm(cand.center - prev))
                    mem = max(0.0, 1.0 - dist / (0.30 * diag))
                    score = 0.55 * score + 0.45 * mem
                cost[i, j] = 1.0 - score

        rows, cols = linear_sum_assignment(cost)
        assigned: dict[str, PersonObs] = {}
        for r, c in zip(rows, cols):
            if cost[r, c] < 0.93:
                role = "red" if c == 0 else "blue"
                assigned[role] = candidates[r]

        for role in ("red", "blue"):
            self.prev_centers[role] = assigned[role].center if role in assigned else None
        return assigned


def normalize_kpts(kpts: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.zeros((N_KPTS, 3), dtype=np.float32)
    n = min(len(kpts), N_KPTS)
    out[:n, 0] = kpts[:n, 0] / max(width, 1)
    out[:n, 1] = kpts[:n, 1] / max(height, 1)
    out[:n, 2] = kpts[:n, 2]
    out[out[:, 2] < 0.05, :2] = 0.0
    return out.reshape(-1)


def normalize_bbox(bbox: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(np.float32)
    return np.array(
        [
            (x1 + x2) * 0.5 / max(width, 1),
            (y1 + y2) * 0.5 / max(height, 1),
            (x2 - x1) / max(width, 1),
            (y2 - y1) / max(height, 1),
        ],
        dtype=np.float32,
    )


def angle_cos(a: np.ndarray | None, b: np.ndarray | None, c: np.ndarray | None) -> float:
    if a is None or b is None or c is None:
        return 0.0
    v1 = a - b
    v2 = c - b
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def dist(a: np.ndarray | None, b: np.ndarray | None, scale: float) -> float:
    if a is None or b is None or scale <= 1e-6:
        return 0.0
    return float(np.linalg.norm(a - b) / scale)


def pair_features(red: PersonObs | None, blue: PersonObs | None, width: int, height: int) -> np.ndarray:
    feats: list[float] = []
    scale = float(np.hypot(width, height))
    roles = [red, blue]
    opponents = [blue, red]
    for me, opp in zip(roles, opponents):
        if me is None or opp is None:
            feats.extend([0.0] * 12)
            continue
        my = me.kpts
        op = opp.kpts
        opp_head = mean_point(op, [NOSE, 1, 2, 3, 4])
        opp_body = mean_point(op, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP])
        left_wrist = safe_point(my, L_WRIST)
        right_wrist = safe_point(my, R_WRIST)
        left_elbow = safe_point(my, L_ELBOW)
        right_elbow = safe_point(my, R_ELBOW)
        left_sh = safe_point(my, L_SHOULDER)
        right_sh = safe_point(my, R_SHOULDER)
        feats.extend(
            [
                dist(left_wrist, opp_head, scale),
                dist(right_wrist, opp_head, scale),
                dist(left_wrist, opp_body, scale),
                dist(right_wrist, opp_body, scale),
                angle_cos(left_sh, left_elbow, left_wrist),
                angle_cos(right_sh, right_elbow, right_wrist),
                float(left_wrist[0] < right_wrist[0]) if left_wrist is not None and right_wrist is not None else 0.0,
                dist(mean_point(my, [L_SHOULDER, R_SHOULDER]), opp_body, scale),
                float(me.colors[0]),
                float(me.colors[1]),
                float(opp.colors[0]),
                float(opp.colors[1]),
            ]
        )
    return np.asarray(feats, dtype=np.float32)


def role_vector(obs: PersonObs | None, width: int, height: int) -> np.ndarray:
    if obs is None:
        return np.zeros(ROLE_DIM, dtype=np.float32)
    return np.concatenate(
        [
            normalize_kpts(obs.kpts, width, height),
            normalize_bbox(obs.bbox, width, height),
            obs.colors.astype(np.float32),
            np.ones(1, dtype=np.float32),
        ]
    ).astype(np.float32)


def frame_feature(assigned: dict[str, PersonObs], width: int, height: int) -> np.ndarray:
    red = assigned.get("red")
    blue = assigned.get("blue")
    return np.concatenate(
        [
            role_vector(red, width, height),
            role_vector(blue, width, height),
            pair_features(red, blue, width, height),
        ]
    ).astype(np.float32)


def add_temporal_deltas(base: np.ndarray) -> np.ndarray:
    if len(base) == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    d1 = np.zeros_like(base)
    d3 = np.zeros_like(base)
    d1[1:] = base[1:] - base[:-1]
    d3[3:] = base[3:] - base[:-3]
    return np.concatenate([base, d1, d3], axis=1).astype(np.float32)

